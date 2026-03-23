from typing import List


def filter_allowed_classes(detections: List[dict], allowed_classes: list[str]) -> List[dict]:
    allowed = set(allowed_classes)
    return [d for d in detections if d["class_name"] in allowed]


def filter_min_area(
    detections: List[dict],
    image_width: int,
    image_height: int,
    min_box_area_norm: float,
) -> List[dict]:
    filtered = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area_norm = (w * h) / (image_width * image_height)

        if area_norm >= min_box_area_norm:
            filtered.append(det)

    return filtered