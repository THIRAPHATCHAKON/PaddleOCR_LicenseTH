import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r""  # Model path
IMG_DIR = r""      # Input image directory
LABEL_DIR = r""    # Output label directory

NUMBER_CONF = 0.50
PROVINCE_CONF = 0.50
IMGSZ = 640

NUMBER_CLS_IDX = 0
PROVINCE_CLS_IDX = 1

detail_model = YOLO(MODEL_PATH)


def find_class_idx(model, target_names):
    for idx, name in model.names.items():
        if name.lower() in target_names:
            return idx
    return None


NUMBER_CLASS_IDX = find_class_idx(detail_model, {"number", "numbers", "plate-number"})
PROVINCE_CLASS_IDX = find_class_idx(detail_model, {"province"})

if NUMBER_CLASS_IDX is None or PROVINCE_CLASS_IDX is None:
    raise ValueError(f"Required classes not found. Available classes: {detail_model.names}")


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def label_image(image_path, label_dir):
    img = cv2.imread(str(image_path))

    if img is None:
        return

    img_h, img_w = img.shape[:2]
    lines = []

    # Detect license plate number
    for r in detail_model.predict(img, conf=NUMBER_CONF, imgsz=IMGSZ, classes=[NUMBER_CLASS_IDX], verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{NUMBER_CLS_IDX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Detect province
    for r in detail_model.predict(img, conf=PROVINCE_CONF, imgsz=IMGSZ, classes=[PROVINCE_CLASS_IDX], verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{PROVINCE_CLS_IDX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_path = label_dir / f"{image_path.stem}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    img_dir = Path(IMG_DIR)
    label_dir = Path(LABEL_DIR)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in image_ext)

    for image_path in files:
        label_image(image_path, label_dir)

    print(f"Completed: {len(files)} images")
