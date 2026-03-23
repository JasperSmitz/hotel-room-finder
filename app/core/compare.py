from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import List, Dict, Tuple

from app.models.schemas import RoomSignature, DetectedObject


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def center_distance(a: DetectedObject, b: DetectedObject) -> float:
    dx = a.center_norm.x - b.center_norm.x
    dy = a.center_norm.y - b.center_norm.y
    return sqrt(dx * dx + dy * dy)


def position_similarity(a: DetectedObject, b: DetectedObject) -> float:
    dist = center_distance(a, b)
    # normalized image-space distance, 0 = same place, ~1.4 = opposite corners
    # map to similarity in a simple way
    return clamp01(1.0 - dist)


def size_similarity(a: DetectedObject, b: DetectedObject) -> float:
    dw = abs(a.size_norm.width - b.size_norm.width)
    dh = abs(a.size_norm.height - b.size_norm.height)
    da = abs(a.size_norm.area - b.size_norm.area)

    # simple blend
    penalty = (dw + dh + da) / 3.0
    return clamp01(1.0 - penalty)


def object_pair_score(a: DetectedObject, b: DetectedObject) -> float:
    emb_sim = cosine_similarity(a.embedding.vector, b.embedding.vector)
    pos_sim = position_similarity(a, b)
    size_sim = size_similarity(a, b)

    # rough initial blend
    return (
        0.60 * emb_sim +
        0.25 * pos_sim +
        0.15 * size_sim
    )


def group_objects_by_class(objects: List[DetectedObject]) -> Dict[str, List[DetectedObject]]:
    grouped: Dict[str, List[DetectedObject]] = defaultdict(list)
    for obj in objects:
        grouped[obj.class_name].append(obj)
    return dict(grouped)


def greedy_match_same_class(
    objects_a: List[DetectedObject],
    objects_b: List[DetectedObject],
) -> List[dict]:
    """
    Greedy same-class matching:
    repeatedly select highest-scoring unmatched pair.
    Good enough for MVP.
    """
    remaining_a = objects_a[:]
    remaining_b = objects_b[:]
    matches: List[dict] = []

    while remaining_a and remaining_b:
        best_score = -1.0
        best_pair: Tuple[int, int] | None = None

        for i, obj_a in enumerate(remaining_a):
            for j, obj_b in enumerate(remaining_b):
                score = object_pair_score(obj_a, obj_b)
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        obj_a = remaining_a.pop(i)
        obj_b = remaining_b.pop(j)

        emb_sim = cosine_similarity(obj_a.embedding.vector, obj_b.embedding.vector)
        pos_sim = position_similarity(obj_a, obj_b)
        siz_sim = size_similarity(obj_a, obj_b)

        matches.append(
            {
                "class_name": obj_a.class_name,
                "object_a": obj_a.object_id,
                "object_b": obj_b.object_id,
                "embedding_similarity": emb_sim,
                "position_similarity": pos_sim,
                "size_similarity": siz_sim,
                "score": best_score,
            }
        )

    return matches


def compare_objects(sig_a: RoomSignature, sig_b: RoomSignature) -> dict:
    grouped_a = group_objects_by_class(sig_a.objects)
    grouped_b = group_objects_by_class(sig_b.objects)

    shared_classes = sorted(set(grouped_a.keys()) & set(grouped_b.keys()))
    matches: List[dict] = []

    for class_name in shared_classes:
        class_matches = greedy_match_same_class(grouped_a[class_name], grouped_b[class_name])
        matches.extend(class_matches)

    if not matches:
        return {
            "score": 0.0,
            "shared_classes": [],
            "matches": [],
        }

    avg_score = sum(m["score"] for m in matches) / len(matches)

    return {
        "score": avg_score,
        "shared_classes": shared_classes,
        "matches": matches,
    }


def relation_to_class_triplet(relation: dict, object_lookup: Dict[str, DetectedObject]) -> str | None:
    subject = object_lookup.get(relation.subject_id)
    obj = object_lookup.get(relation.object_id)

    if not subject or not obj:
        return None

    return f"{subject.class_name}|{relation.predicate}|{obj.class_name}"


def compare_relations(sig_a: RoomSignature, sig_b: RoomSignature) -> dict:
    lookup_a = {obj.object_id: obj for obj in sig_a.objects}
    lookup_b = {obj.object_id: obj for obj in sig_b.objects}

    rels_a = set()
    rels_b = set()

    for rel in sig_a.relations:
        triplet = relation_to_class_triplet(rel, lookup_a)
        if triplet:
            rels_a.add(triplet)

    for rel in sig_b.relations:
        triplet = relation_to_class_triplet(rel, lookup_b)
        if triplet:
            rels_b.add(triplet)

    if not rels_a and not rels_b:
        return {
            "score": 1.0,
            "shared": [],
            "only_a": [],
            "only_b": [],
        }

    if not rels_a or not rels_b:
        return {
            "score": 0.0,
            "shared": [],
            "only_a": sorted(rels_a),
            "only_b": sorted(rels_b),
        }

    shared = rels_a & rels_b
    union = rels_a | rels_b
    score = len(shared) / len(union) if union else 0.0

    return {
        "score": score,
        "shared": sorted(shared),
        "only_a": sorted(rels_a - rels_b),
        "only_b": sorted(rels_b - rels_a),
    }


def compare_signatures(sig_a: RoomSignature, sig_b: RoomSignature) -> dict:
    global_sim = cosine_similarity(sig_a.embedding.vector, sig_b.embedding.vector)
    object_result = compare_objects(sig_a, sig_b)
    relation_result = compare_relations(sig_a, sig_b)

    final_score = (
        0.40 * global_sim +
        0.45 * object_result["score"] +
        0.15 * relation_result["score"]
    )

    return {
        "image_a": sig_a.image.source,
        "image_b": sig_b.image.source,
        "global_similarity": global_sim,
        "object_score": object_result["score"],
        "relation_score": relation_result["score"],
        "final_score": final_score,
        "shared_classes": object_result["shared_classes"],
        "object_matches": object_result["matches"],
        "relation_overlap": relation_result,
        "summary": build_summary(global_sim, object_result["score"], relation_result["score"], final_score),
    }


def build_summary(global_sim: float, object_score: float, relation_score: float, final_score: float) -> str:
    if final_score >= 0.80:
        overall = "Very strong match"
    elif final_score >= 0.65:
        overall = "Strong match"
    elif final_score >= 0.50:
        overall = "Moderate match"
    else:
        overall = "Weak match"

    return (
        f"{overall}. "
        f"Global={global_sim:.3f}, "
        f"Objects={object_score:.3f}, "
        f"Relations={relation_score:.3f}, "
        f"Final={final_score:.3f}."
    )