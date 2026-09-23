import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r""   # Model path
IMG_DIR = r""      # Input image directory
OUTPUT_DIR = r""   # Output crop directory

PLATE_CONF = 0.50
IMGSZ = 640

model = YOLO(MODEL_PATH)


def find_class_idx(model, target_names):
    for idx, name in model.names.items():
        if name.lower() in target_names:
            return idx
    return None


PLATE_CLASS_IDX = find_class_idx(model, {"license-plate", "license_plate", "plate"})

if PLATE_CLASS_IDX is None:
    raise ValueError(f"Required class not found. Available classes: {model.names}")


def crop_plate(image_path, output_dir):
    img = cv2.imread(str(image_path))

    if img is None:
        return

    img_h, img_w = img.shape[:2]
    plate_count = 0

    # Detect license plates
    results = model.predict(img, conf=PLATE_CONF, imgsz=IMGSZ, classes=[PLATE_CLASS_IDX], verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Keep bounding box inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop license plate
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            plate_count += 1
            output_path = output_dir / f"{image_path.stem}_plate_{plate_count}.jpg"
            cv2.imwrite(str(output_path), crop)


if __name__ == "__main__":
    img_dir = Path(IMG_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in image_ext)

    for image_path in files:
        crop_plate(image_path, output_dir)

    print(f"Completed: {len(files)} images")
