import os
import cv2
import numpy as np
from PIL import Image


def draw_detections(image: Image.Image, detections: list, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img)
    return output_path