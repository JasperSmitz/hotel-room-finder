from pydantic import BaseModel


class Settings(BaseModel):
    detector_model: str = "yolo11m.pt"
    detector_confidence: float = 0.25
    max_detections: int = 25

    embedder_model_name: str = "ViT-B-32"
    embedder_pretrained: str = "laion2b_s34b_b79k"

    crop_margin: float = 0.10
    save_debug_images: bool = True
    debug_output_dir: str = "debug"


settings = Settings()