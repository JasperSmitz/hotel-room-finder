import os
from PIL import Image
from collections import defaultdict
from app.core.relations import build_relations
from app.core.postprocess import filter_allowed_classes, filter_min_area
from app.core.dedupe import dedupe_same_class_detections
from app.config import settings
from app.core.crops import expand_bbox, crop_image
from app.core.draw import draw_detections
from app.core.geometry import normalize_bbox, bbox_center_norm, bbox_size_norm
from app.models.schemas import (
    RoomSignature,
    ImageInfo,
    EmbeddingInfo,
    DetectedObject,
    DebugInfo,
    BBoxPx,
)
from app.core.detector import ObjectDetector
from app.core.embedder import ImageEmbedder
from app.core.crop_store import save_crop


class SignaturePipeline:
    def __init__(self):
        self.detector = ObjectDetector(
            model_name=settings.detector_model,
            confidence=settings.detector_confidence,
            max_detections=settings.max_detections,
        )
        self.embedder = ImageEmbedder(
            model_name=settings.embedder_model_name,
            pretrained=settings.embedder_pretrained,
        )

    def build_from_path(self, image_path: str, save_debug_image: bool = True) -> RoomSignature:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        global_vector = self.embedder.embed_image(image)
        detections = self.detector.detect(image)
        detections = filter_allowed_classes(detections, settings.allowed_classes)
        detections = filter_min_area(detections, width, height, settings.min_box_area_norm)
        detections = dedupe_same_class_detections(
            detections,
            image_width=width,
            image_height=height,
            iou_threshold=settings.dedupe_iou_threshold,
            center_distance_threshold=settings.dedupe_center_distance_threshold,
        )

        annotated_path = None
        if save_debug_image and settings.save_debug_images:
            filename = os.path.basename(image_path)
            stem, _ = os.path.splitext(filename)
            annotated_path = os.path.join(settings.debug_output_dir, f"{stem}_debug.jpg")
            draw_detections(image, detections, annotated_path)

        class_counts = defaultdict(int)
        objects = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            ex1, ey1, ex2, ey2 = expand_bbox(
                x1, y1, x2, y2, width, height, settings.crop_margin
            )
            crop = crop_image(image, ex1, ey1, ex2, ey2)
            crop_path = None
            if settings.save_object_crops:
                filename = os.path.basename(image_path)
                stem, _ = os.path.splitext(filename)
                crop_filename = f"{stem}_{object_id}.jpg"
                crop_path = os.path.join(settings.crop_output_dir, crop_filename)
                save_crop(crop, crop_path)
            crop_vector = self.embedder.embed_image(crop)

            bbox_norm = normalize_bbox(x1, y1, x2, y2, width, height)
            center_norm = bbox_center_norm(bbox_norm)
            size_norm = bbox_size_norm(bbox_norm)

            class_name = det["class_name"]
            object_index = class_counts[class_name]
            object_id = f"{class_name}_{object_index}"
            class_counts[class_name] += 1

            objects.append(
                DetectedObject(
                    object_id=object_id,
                    class_name=class_name,
                    confidence=det["confidence"],
                    bbox_px=BBoxPx(x1=x1, y1=y1, x2=x2, y2=y2),
                    bbox_norm=bbox_norm,
                    center_norm=center_norm,
                    size_norm=size_norm,
                    crop_margin=settings.crop_margin,
                    crop_path=crop_path,
                    embedding=EmbeddingInfo(
                        model="open_clip",
                        model_name=settings.embedder_model_name,
                        dim=len(crop_vector),
                        vector=crop_vector,
                    ),
                )
            )

        relations = build_relations(objects)

        return RoomSignature(
            image=ImageInfo(source=image_path, width=width, height=height),
            embedding=EmbeddingInfo(
                model="open_clip",
                model_name=settings.embedder_model_name,
                dim=len(global_vector),
                vector=global_vector,
            ),
            objects=objects,
            relations=relations,
            debug=DebugInfo(annotated_image_path=annotated_path),
        )