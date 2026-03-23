from math import sqrt
from typing import List

from app.models.schemas import DetectedObject, ObjectRelation


LEFT_RIGHT_THRESHOLD = 0.08
ABOVE_BELOW_THRESHOLD = 0.08
NEAR_DISTANCE_THRESHOLD = 0.35
ALIGN_THRESHOLD = 0.10


def normalized_distance(a: DetectedObject, b: DetectedObject) -> float:
    dx = a.center_norm.x - b.center_norm.x
    dy = a.center_norm.y - b.center_norm.y
    return sqrt(dx * dx + dy * dy)


def build_relations(objects: List[DetectedObject]) -> List[ObjectRelation]:
    relations: List[ObjectRelation] = []

    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue

            a = objects[i]
            b = objects[j]

            dx = b.center_norm.x - a.center_norm.x
            dy = b.center_norm.y - a.center_norm.y
            dist = normalized_distance(a, b)

            if dx > LEFT_RIGHT_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="left_of",
                        object_id=b.object_id,
                        score=min(1.0, abs(dx)),
                    )
                )
            elif dx < -LEFT_RIGHT_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="right_of",
                        object_id=b.object_id,
                        score=min(1.0, abs(dx)),
                    )
                )

            if dy > ABOVE_BELOW_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="above",
                        object_id=b.object_id,
                        score=min(1.0, abs(dy)),
                    )
                )
            elif dy < -ABOVE_BELOW_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="below",
                        object_id=b.object_id,
                        score=min(1.0, abs(dy)),
                    )
                )

            if dist < NEAR_DISTANCE_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="near",
                        object_id=b.object_id,
                        score=max(0.0, 1.0 - (dist / NEAR_DISTANCE_THRESHOLD)),
                    )
                )

            if abs(a.center_norm.y - b.center_norm.y) < ALIGN_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="horizontally_aligned_with",
                        object_id=b.object_id,
                        score=max(0.0, 1.0 - abs(a.center_norm.y - b.center_norm.y) / ALIGN_THRESHOLD),
                    )
                )

            if abs(a.center_norm.x - b.center_norm.x) < ALIGN_THRESHOLD:
                relations.append(
                    ObjectRelation(
                        subject_id=a.object_id,
                        predicate="vertically_aligned_with",
                        object_id=b.object_id,
                        score=max(0.0, 1.0 - abs(a.center_norm.x - b.center_norm.x) / ALIGN_THRESHOLD),
                    )
                )

    return relations