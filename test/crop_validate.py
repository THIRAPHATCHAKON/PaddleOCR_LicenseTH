# from ultralytics import YOLO
# import cv2
# from pathlib import Path

# # =========================
# # CONFIG
# # =========================
# MODEL_PATH = "final_320.pt"
# IMAGE_PATH = "output_enhanced.jpg"

# OUTPUT_DIR = Path("crop_plate")
# OUTPUT_DIR.mkdir(exist_ok=True)

# CONF_THRESHOLD = 0.70

# # ชื่อ class ป้ายทะเบียนในโมเดล
# PLATE_CLASS_NAME = "license-plate"


# # =========================
# # LOAD MODEL
# # =========================
# model = YOLO(MODEL_PATH)

# print("Classes:", model.names)


# # =========================
# # LOAD IMAGE
# # =========================
# image = cv2.imread(IMAGE_PATH)

# if image is None:
#     raise FileNotFoundError(
#         f"ไม่พบรูป: {IMAGE_PATH}"
#     )


# # =========================
# # YOLO DETECTION
# # =========================
# results = model(
#     IMAGE_PATH,
#     conf=CONF_THRESHOLD,
#     verbose=False
# )


# # =========================
# # CROP LICENSE PLATE
# # =========================
# plate_count = 0

# for result in results:

#     for box in result.boxes:

#         # -------------------------
#         # Class
#         # -------------------------
#         class_id = int(box.cls[0])
#         class_name = model.names[class_id]

#         # เอาเฉพาะ license-plate
#         if class_name != PLATE_CLASS_NAME:
#             continue

#         # -------------------------
#         # Confidence
#         # -------------------------
#         confidence = float(box.conf[0])
#         confidence_percent = confidence * 100

#         # -------------------------
#         # Bounding Box
#         # -------------------------
#         x1, y1, x2, y2 = map(
#             int,
#             box.xyxy[0].tolist()
#         )

#         # กันพิกัดเกินภาพ
#         h, w = image.shape[:2]

#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(w, x2)
#         y2 = min(h, y2)

#         # -------------------------
#         # Crop
#         # -------------------------
#         plate_crop = image[
#             y1:y2,
#             x1:x2
#         ].copy()

#         if plate_crop.size == 0:
#             continue

#         plate_count += 1

#         # -------------------------
#         # Save
#         # -------------------------
#         output_path = (
#             OUTPUT_DIR /
#             f"plate_{plate_count}.jpg"
#         )

#         cv2.imwrite(
#             str(output_path),
#             plate_crop
#         )

#         print(
#             f"Plate {plate_count}: "
#             f"{confidence_percent:.2f}%"
#         )

#         print(
#             f"Saved: {output_path}"
#         )


# # =========================
# # RESULT
# # =========================
# print("\n=========================")

# if plate_count == 0:
#     print("ไม่พบป้ายทะเบียน")
# else:
#     print(
#         f"พบป้ายทะเบียนทั้งหมด {plate_count} ป้าย"
#     )

# print(
#     f"Output: {OUTPUT_DIR.resolve()}"
# )

# print("=========================")

from ultralytics import YOLO
import cv2
from pathlib import Path

# =========================
# CONFIG
# =========================
MODEL_PATH = "best.pt"
IMAGE_PATH = "plate_perspective.jpg"

OUTPUT_DIR = Path("crop_plate")
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.60

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

print("Classes:", model.names)

# =========================
# LOAD IMAGE
# =========================
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"ไม่พบรูป: {IMAGE_PATH}"
    )

h, w = image.shape[:2]

# =========================
# DETECT
# =========================
results = model(
    IMAGE_PATH,
    conf=CONF_THRESHOLD,
    verbose=False
)

# =========================
# COUNTER
# =========================
number_count = 0
province_count = 0

# =========================
# PROCESS
# =========================
for result in results:

    for box in result.boxes:

        # -------------------------
        # CLASS
        # -------------------------
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # เอาเฉพาะ number และ province
        if class_name not in ["number", "province"]:
            continue

        # -------------------------
        # CONFIDENCE
        # -------------------------
        confidence = float(box.conf[0])
        confidence_percent = confidence * 100

        # -------------------------
        # BOUNDING BOX
        # -------------------------
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # ป้องกันพิกัดเกินภาพ
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # -------------------------
        # CROP
        # -------------------------
        crop = image[
            y1:y2,
            x1:x2
        ].copy()

        if crop.size == 0:
            continue

        # =========================
        # NUMBER
        # =========================
        if class_name == "number":

            number_count += 1

            output_path = (
                OUTPUT_DIR /
                f"number_{number_count}.jpg"
            )

        # =========================
        # PROVINCE
        # =========================
        elif class_name == "province":

            province_count += 1

            output_path = (
                OUTPUT_DIR /
                f"province_{province_count}.jpg"
            )

        # =========================
        # SAVE
        # =========================
        cv2.imwrite(
            str(output_path),
            crop
        )

        print(
            f"{class_name:<10}"
            f" | Confidence: {confidence_percent:.2f}%"
            f" | Box: ({x1},{y1}) -> ({x2},{y2})"
        )

        print(
            f"Saved -> {output_path}"
        )


# =========================
# RESULT
# =========================
print("\n=========================")

print(
    f"Number   : {number_count}"
)

print(
    f"Province : {province_count}"
)

print(
    f"Output   : {OUTPUT_DIR.resolve()}"
)

print("=========================")