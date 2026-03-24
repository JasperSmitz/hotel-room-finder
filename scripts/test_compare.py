import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from app.models.schemas import RoomSignature
from app.core.compare import compare_signatures


def load_signature(path: str) -> RoomSignature:
    with open(path, "r", encoding="utf-16") as f:
        data = json.load(f)
    return RoomSignature(**data)


if __name__ == "__main__":
    sig_a = load_signature("data/output/result_haarlem1.json")
    sig_b = load_signature("data/output/result_haarlem2.json")

    result = compare_signatures(sig_a, sig_b)
    print(json.dumps(result, indent=2))