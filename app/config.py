from pydantic import BaseModel, Field


class Settings(BaseModel):
    detector_model: str = "yolov8n.pt"
    detector_confidence: float = 0.35
    max_detections: int = 25

    allowed_classes: list[str] = Field(default_factory=lambda: [
        "bed",
        "chair",
        "couch",
        "tv",
        "potted plant",
        "dining table",
        "sink",
        "toilet",
        "vase",
    ])

    min_box_area_norm: float = 0.01

    embedder_model_name: str = "ViT-B-32"
    embedder_pretrained: str = "laion2b_s34b_b79k"

    crop_margin: float = 0.10
    save_debug_images: bool = True
    debug_output_dir: str = "debug"

    dedupe_iou_threshold: float = 0.80
    dedupe_center_distance_threshold: float = 0.08 


settings = Settings()