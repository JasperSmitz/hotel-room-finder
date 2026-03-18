from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image
import numpy as np


class ObjectDetector:
    def __init__(self, model_name: str, confidence: float, max_detections: int):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.max_detections = max_detections

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        img = np.array(image.convert("RGB"))
        results = self.model.predict(
            source=img,
            conf=self.confidence,
            max_det=self.max_detections,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                x1, y1, x2, y2 = [int(v) for v in xyxy]

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return detections