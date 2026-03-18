from app.models.schemas import BBoxNorm, PointNorm, SizeNorm


def normalize_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> BBoxNorm:
    return BBoxNorm(
        x1=x1 / width,
        y1=y1 / height,
        x2=x2 / width,
        y2=y2 / height,
    )


def bbox_center_norm(bbox: BBoxNorm) -> PointNorm:
    return PointNorm(
        x=(bbox.x1 + bbox.x2) / 2.0,
        y=(bbox.y1 + bbox.y2) / 2.0,
    )


def bbox_size_norm(bbox: BBoxNorm) -> SizeNorm:
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    return SizeNorm(
        width=w,
        height=h,
        area=w * h,
    )