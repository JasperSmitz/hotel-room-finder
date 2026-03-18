from PIL import Image


def expand_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, margin: float):
    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin)
    my = int(bh * margin)

    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(img_w, x2 + mx)
    ny2 = min(img_h, y2 + my)

    return nx1, ny1, nx2, ny2


def crop_image(image: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    return image.crop((x1, y1, x2, y2))