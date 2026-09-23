from pathlib import Path

IMG_DIR = r"D:\license_plate_and_car_detection\crop_plate_dataset\output_2"
OUTPUT_TXT = r"D:\license_plate_and_car_detection\train.txt"

TXT_IMAGE_PREFIX = "img/train"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

img_dir = Path(IMG_DIR)
images = sorted(
    (file for file in img_dir.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS),
    key=lambda x: x.name
)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for image in images:
        f.write(f"{TXT_IMAGE_PREFIX}/{image.name}\t\n")

print(f"Completed: {len(images)} images")