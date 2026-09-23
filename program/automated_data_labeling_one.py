import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model_path = "" # Path
model = YOLO(str(model_path))

def find_class_idx(model_names: dict, target_name: str):
    return next(
        (idx for idx, name in model_names.items() if name.lower() == target_name.lower()),
        None
    )

CAR_CLASS_IDX = find_class_idx(model.names, "car")
PLATE_CLASS_IDX = find_class_idx(model.names, "license-plate")

if CAR_CLASS_IDX is None:
    raise ValueError(
        f"ไม่พบ class 'car' ในโมเดล\n"
        f"classes ที่มี: {model.names}\n"
    )
if PLATE_CLASS_IDX is None:
    raise ValueError(
        f"ไม่พบ class 'license-plate' ในโมเดล\n"
        f"classes ที่มี: {model.names}\n"
    )

CONF_BY_CLASS = {
    CAR_CLASS_IDX: 0.70,
    PLATE_CLASS_IDX: 0.80,
}
MIN_CONF_FOR_PREDICT = min(CONF_BY_CLASS.values())

IMG_DIR   = r"" #Path Input
LABEL_DIR = r"" #Path OutPut

IMG_SIZE = 1280

CLASS_NAMES   = ["car", "license-plate"]
CAR_CLS_IDX   = 0
PLATE_CLS_IDX = 1

MODEL_IDX_TO_OUTPUT_IDX = {
    CAR_CLASS_IDX: CAR_CLS_IDX,
    PLATE_CLASS_IDX: PLATE_CLS_IDX,
}


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def label_image(image_path: Path, label_dir: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"อ่านไม่ได้: {image_path.name}")
        return 0, 0

    img_h, img_w = img.shape[:2]
    lines = []
    car_count = plate_count = 0

    results = model.predict(
        img,
        conf=MIN_CONF_FOR_PREDICT,
        imgsz=IMG_SIZE,
        classes=[CAR_CLASS_IDX, PLATE_CLASS_IDX],
        verbose=False,
    )

    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_BY_CLASS.get(cls_idx, 1.0):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)

            output_idx = MODEL_IDX_TO_OUTPUT_IDX[cls_idx]
            lines.append(f"{output_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if cls_idx == CAR_CLASS_IDX:
                car_count += 1
            elif cls_idx == PLATE_CLASS_IDX:
                plate_count += 1

    out_path = label_dir / (image_path.stem + ".txt")
    out_path.write_text("\n".join(lines), encoding="utf-8")

    return car_count, plate_count

if __name__ == "__main__":
    img_dir   = Path(IMG_DIR)
    label_dir = Path(LABEL_DIR)
    label_dir.mkdir(parents=True, exist_ok=True)

    classes_path = label_dir / "classes.txt"
    classes_path.write_text("\n".join(CLASS_NAMES), encoding="utf-8")
    print(f"classes.txt → {classes_path}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {i}: {name}")
    print()

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    print(f"พบ {len(files)} ภาพใน {img_dir}\n")

    total_car = total_plate = 0

    for i, f in enumerate(files, 1):
        cars, plates = label_image(f, label_dir)
        total_car   += cars
        total_plate += plates
        print(f"[{i:>4}/{len(files)}] {f.name:<60} car={cars}  plate={plates}")
        
    print("Ok!")
    print(f"car             : {total_car}")
    print(f"plate           : {total_plate}")
    print(f"label           : {label_dir}")
    print(f"classes.txt     : {classes_path}")