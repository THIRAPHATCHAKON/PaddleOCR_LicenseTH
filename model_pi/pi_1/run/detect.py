import cv2
from ultralytics import YOLO
from ocr import run_ocr


USE_CLASSES = [0, 3]  # car + license_plate


class PlateDetector:
    def __init__(self, model_path, conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, img_path):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError("Image not found")

        results = self.model(img)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # 🔥 filter class
            if cls not in USE_CLASSES:
                continue

            if conf < self.conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # กันหลุดภาพ
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]

            ocr_text, ocr_score, plate, valid = run_ocr(crop)

            detections.append({
                "class_id": cls,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "ocr_raw": ocr_text,
                "ocr_score": ocr_score,
                "plate": plate,
                "valid": valid
            })

        return img, detections