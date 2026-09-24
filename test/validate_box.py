# from ultralytics import YOLO
# import cv2
# from pathlib import Path

# # =========================
# # CONFIG
# # =========================
# MODEL_PATH = "final_320.pt"
# IMAGE_PATH = "output_enhanced.jpg"

# OUTPUT_DIR = Path("crop_results")
# OUTPUT_DIR.mkdir(exist_ok=True)

# CONF_THRESHOLD = 0.30

# # =========================
# # LOAD MODEL
# # =========================
# model = YOLO(MODEL_PATH)

# # =========================
# # DETECT
# # =========================
# results = model(
#     IMAGE_PATH,
#     conf=CONF_THRESHOLD
# )

# image = cv2.imread(IMAGE_PATH)

# # =========================
# # PROCESS RESULT
# # =========================
# for result in results:

#     for i, box in enumerate(result.boxes):

#         # Bounding box
#         x1, y1, x2, y2 = map(
#             int,
#             box.xyxy[0].tolist()
#         )

#         # Confidence
#         confidence = float(box.conf[0])

#         # Class ID
#         class_id = int(box.cls[0])

#         # Class name
#         class_name = model.names[class_id]

#         # แปลงเป็น %
#         confidence_percent = confidence * 100

#         print(
#             f"{class_name} : "
#             f"{confidence_percent:.2f}%"
#         )

#         # =========================
#         # CROP
#         # =========================
#         crop = image[y1:y2, x1:x2].copy()

#         if crop.size == 0:
#             continue

#         # =========================
#         # LABEL
#         # =========================
#         label = (
#             f"{class_name} "
#             f"{confidence_percent:.2f}%"
#         )

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.6
#         thickness = 2

#         # วาดกรอบรอบ Crop
#         cv2.rectangle(
#             crop,
#             (0, 0),
#             (crop.shape[1] - 1, crop.shape[0] - 1),
#             (0, 255, 0),
#             2
#         )

#         # พื้นหลัง Label
#         (text_w, text_h), _ = cv2.getTextSize(
#             label,
#             font,
#             font_scale,
#             thickness
#         )

#         cv2.rectangle(
#             crop,
#             (0, 0),
#             (text_w + 10, text_h + 12),
#             (0, 255, 0),
#             -1
#         )

#         # เขียน Label
#         cv2.putText(
#             crop,
#             label,
#             (5, text_h + 5),
#             font,
#             font_scale,
#             (0, 0, 0),
#             thickness,
#             cv2.LINE_AA
#         )

#         # =========================
#         # SAVE
#         # =========================
#         output_path = (
#             OUTPUT_DIR /
#             f"{class_name}_{i}_{confidence_percent:.2f}.jpg"
#         )

#         cv2.imwrite(
#             str(output_path),
#             crop
#         )

#         # แสดง Crop
#         cv2.imshow(
#             f"{class_name} {confidence_percent:.2f}%",
#             crop
#         )


# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("เสร็จแล้ว")

from ultralytics import YOLO
import cv2
from pathlib import Path

# =========================
# CONFIG
# =========================
MODEL_PATH = "final_320.pt"
IMAGE_PATH = "output_enhanced.jpg"

OUTPUT_PATH = "detection_result.jpg"

CONF_THRESHOLD = 0.70

# ขนาด Label
FONT_SCALE = 1.5
FONT_THICKNESS = 3
BOX_THICKNESS = 3
PADDING = 10

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# DETECT
# =========================
results = model(
    IMAGE_PATH,
    conf=CONF_THRESHOLD
)

# =========================
# LOAD IMAGE
# =========================
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"ไม่พบรูป: {IMAGE_PATH}"
    )

img_h, img_w = image.shape[:2]

# =========================
# PROCESS RESULT
# =========================
for result in results:

    for box in result.boxes:

        # Bounding Box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Confidence
        confidence = float(box.conf[0])
        confidence_percent = confidence * 100

        # Class
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print(
            f"{class_name:<15} : "
            f"{confidence_percent:.2f}%"
        )

        # =========================
        # LABEL
        # =========================
        label = (
            f"{class_name} "
            f"{confidence_percent:.2f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        # =========================
        # DRAW BOUNDING BOX
        # =========================
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            BOX_THICKNESS
        )

        # =========================
        # TEXT SIZE
        # =========================
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            FONT_SCALE,
            FONT_THICKNESS
        )

        label_width = text_w + (PADDING * 2)
        label_height = text_h + baseline + (PADDING * 2)

        # =========================
        # LABEL ด้านล่างกรอบ
        # =========================
        label_x1 = x1
        label_y1 = y2

        label_x2 = min(
            label_x1 + label_width,
            img_w - 1
        )

        label_y2 = min(
            label_y1 + label_height,
            img_h - 1
        )

        # ถ้าด้านล่างไม่มีพื้นที่
        # ให้เอา Label เข้าไปอยู่ด้านล่างภายในกรอบ
        if y2 + label_height >= img_h:

            label_y2 = y2

            label_y1 = max(
                0,
                y2 - label_height
            )

        # =========================
        # LABEL BACKGROUND
        # =========================
        cv2.rectangle(
            image,
            (label_x1, label_y1),
            (label_x2, label_y2),
            (0, 255, 0),
            -1
        )

        # =========================
        # TEXT
        # =========================
        text_x = label_x1 + PADDING

        text_y = (
            label_y1
            + PADDING
            + text_h
        )

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS,
            cv2.LINE_AA
        )


# =========================
# SAVE
# =========================
success = cv2.imwrite(
    OUTPUT_PATH,
    image
)

if success:

    print(
        f"\nSaved: {Path(OUTPUT_PATH).resolve()}"
    )

else:

    print("\nSave failed")


print("=========================")
print("เสร็จแล้ว")
print("=========================")