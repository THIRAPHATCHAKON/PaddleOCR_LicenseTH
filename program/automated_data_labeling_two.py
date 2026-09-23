import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

plate_model_path = "" #Path Model
car_model_path = "" #Path Model
plate_model = YOLO(str(plate_model_path))
car_model   = YOLO(str(car_model_path))

print(f"Plate model classes : {plate_model.names}")
print(f"Car   model classes : {car_model.names}")

CAR_CLASS_IDX = next(
    (idx for idx, name in car_model.names.items() if name.lower() == "car"),
    None
)
if CAR_CLASS_IDX is None:
    raise ValueError(
        f"ไม่พบ class 'car' ในโมเดล car_model\n"
        f"classes ที่มี: {car_model.names}\n"
        f"กรุณาเปลี่ยนชื่อ class หรือแก้ไข CAR_CLASS_IDX ด้วยตนเอง"
    )
print(f"ใช้ class index {CAR_CLASS_IDX} = '{car_model.names[CAR_CLASS_IDX]}' สำหรับ car")

CAR_CONF   = 0.70
PLATE_CONF = 0.80

IMG_DIR   = r"" #Path Input
LABEL_DIR = r"" #Path OutPut

CLASS_NAMES   = ["car", "license-plate"]
CAR_CLS_IDX   = 0
PLATE_CLS_IDX = 1


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

    for r in car_model.predict(img, conf=CAR_CONF, classes=[CAR_CLASS_IDX], verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{CAR_CLS_IDX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            car_count += 1

    for r in plate_model.predict(img, conf=PLATE_CONF, imgsz=1280, verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{PLATE_CLS_IDX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            plate_count += 1

    # บันทึก .txt
    out_path = label_dir / (image_path.stem + ".txt")
    out_path.write_text("\n".join(lines), encoding="utf-8")

    return car_count, plate_count

if __name__ == "__main__":
    img_dir   = Path(IMG_DIR)
    label_dir = Path(LABEL_DIR)
    label_dir.mkdir(parents=True, exist_ok=True)

    # สร้าง classes.txt
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