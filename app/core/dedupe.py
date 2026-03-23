from math import sqrt
from typing import List, Dict, Any


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def normalized_center_distance(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    dx = (ax - bx) / image_width
    dy = (ay - by) / image_height
    return sqrt(dx * dx + dy * dy)


def dedupe_same_class_detections(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.80,
    center_distance_threshold: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    Conservative dedupe:
    remove only likely duplicate detections of the same physical object.
    """

    if not detections:
        return detections

    sorted_detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []

    for candidate in sorted_detections:
        candidate_box = candidate["bbox"]
        candidate_class = candidate["class_name"]

        is_duplicate = False

        for existing in kept:
            if existing["class_name"] != candidate_class:
                continue

            existing_box = existing["bbox"]

            iou = bbox_iou(candidate_box, existing_box)
            center_dist = normalized_center_distance(
                candidate_box,
                existing_box,
                image_width,
                image_height,
            )

            if iou >= iou_threshold and center_dist <= center_distance_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept