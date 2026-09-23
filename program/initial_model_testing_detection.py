import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model_path = "" # Path InPut
model = YOLO(str(model_path))

print(f"   Model classes : {model.names}")

def find_class_idx(model_names: dict, target_name: str):
    return next(
        (idx for idx, name in model_names.items() if name.lower() == target_name.lower()),
        None
    )

PLATE_CLASS_IDX = find_class_idx(model.names, "license-plate")

if PLATE_CLASS_IDX is None:
    raise ValueError(
        f"not find class 'license-plate' in model\n"
        f"classes ที่มี: {model.names}\n"
        f"กรุณาเปลี่ยนชื่อ class หรือแก้ไข PLATE_CLASS_IDX ด้วยตนเอง"
    )

print(f"class index {PLATE_CLASS_IDX} = '{model.names[PLATE_CLASS_IDX]}' สำหรับ license-plate")

CONF_THRESHOLD = 0.80

IMG_DIR  = r"D:\license_plate_and_car_detection\evaluation_ocr"
CROP_DIR = r"D:\license_plate_and_car_detection\evaluation_ocr\plate"

IMG_SIZE = 1280

PAD = 0

def crop_plates(image_path: Path, crop_dir: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"not read: {image_path.name}")
        return 0

    img_h, img_w = img.shape[:2]

    results = model.predict(
        img,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        classes=[PLATE_CLASS_IDX],
        verbose=False,
    )

    plate_count = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            x1 = max(0, x1 - PAD)
            y1 = max(0, y1 - PAD)
            x2 = min(img_w, x2 + PAD)
            y2 = min(img_h, y2 + PAD)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]

            plate_count += 1

            if plate_count == 1:
                out_name = f"{image_path.stem}.jpg"
            else:
                out_name = f"{image_path.stem}_{plate_count}.jpg"

            out_path = crop_dir / out_name
            cv2.imwrite(str(out_path), crop)

    return plate_count


if __name__ == "__main__":
    img_dir  = Path(IMG_DIR)
    crop_dir = Path(CROP_DIR)
    crop_dir.mkdir(parents=True, exist_ok=True)

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    print(f"find {len(files)} in {img_dir}\n")

    total_plate = 0
    images_with_plate = 0

    for i, f in enumerate(files, 1):
        plates = crop_plates(f, crop_dir)
        total_plate += plates
        if plates > 0:
            images_with_plate += 1
        print(f"[{i:>4}/{len(files)}] {f.name:<60} plate={plates}")

    print(f"\nOk!")
    print(f"ภาพทั้งหมด         : {len(files)}")
    print(f"ภาพที่เจอป้าย       : {images_with_plate}")
    print(f"รวมป้ายที่ครอปได้    : {total_plate}")
    print(f"บันทึกที่           : {crop_dir}")