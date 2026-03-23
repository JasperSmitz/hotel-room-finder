from typing import List, Optional
from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    source: str
    width: int
    height: int


class EmbeddingInfo(BaseModel):
    model: str
    model_name: str
    dim: int
    vector: List[float]


class BBoxPx(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class BBoxNorm(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PointNorm(BaseModel):
    x: float
    y: float


class SizeNorm(BaseModel):
    width: float
    height: float
    area: float


class ObjectRelation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    score: float


class DetectedObject(BaseModel):
    object_id: str
    class_name: str
    confidence: float
    bbox_px: BBoxPx
    bbox_norm: BBoxNorm
    center_norm: PointNorm
    size_norm: SizeNorm
    crop_margin: float
    embedding: EmbeddingInfo


class DebugInfo(BaseModel):
    annotated_image_path: Optional[str] = None


class RoomSignature(BaseModel):
    image: ImageInfo
    embedding: EmbeddingInfo
    objects: List[DetectedObject] = Field(default_factory=list)
    relations: List[ObjectRelation] = Field(default_factory=list)
    debug: DebugInfo


class SignatureFromPathRequest(BaseModel):
    image_path: str
    save_debug_image: bool = True