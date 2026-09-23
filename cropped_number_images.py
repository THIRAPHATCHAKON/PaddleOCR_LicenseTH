import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r""   # Model path
IMG_DIR = r""      # Input image directory
OUTPUT_DIR = r""   # Output crop directory

NUMBER_CONF = 0.50
PROVINCE_CONF = 0.50
IMGSZ = 640

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


def crop_image(image_path, output_dir):
    img = cv2.imread(str(image_path))

    if img is None:
        return

    img_h, img_w = img.shape[:2]
    number_count = 0
    province_count = 0

    # Detect number and province
    results = detail_model.predict(
        img,
        conf=min(NUMBER_CONF, PROVINCE_CONF),
        imgsz=IMGSZ,
        classes=[NUMBER_CLASS_IDX, PROVINCE_CLASS_IDX],
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Keep bounding box inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Save number
            if cls_idx == NUMBER_CLASS_IDX and confidence >= NUMBER_CONF:
                number_count += 1
                output_path = output_dir / f"{image_path.stem}_number_{number_count}.jpg"
                cv2.imwrite(str(output_path), crop)

            # Save province
            elif cls_idx == PROVINCE_CLASS_IDX and confidence >= PROVINCE_CONF:
                province_count += 1
                output_path = output_dir / f"{image_path.stem}_province_{province_count}.jpg"
                cv2.imwrite(str(output_path), crop)


if __name__ == "__main__":
    img_dir = Path(IMG_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in image_ext)

    for image_path in files:
        crop_image(image_path, output_dir)

    print(f"Completed: {len(files)} images")
