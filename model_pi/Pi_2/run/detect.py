# import cv2
# from ultralytics import YOLO
# from plate_ocr import read_plate_array

# USE_CLASSES = [0, 1]   # 0 = car, 1 = license_plate
# PLATE_CLASS_ID = 1     # class id ของป้ายทะเบียน (ใช้ค่านี้ที่เดียว ไม่ hardcode ซ้ำที่อื่น)


# class PlateDetector:
#     def __init__(self, model_path, conf=0.3,
#                  split_ratio=0.60, province_threshold=70, ocr_conf_threshold=0.6):
#         self.model = YOLO(model_path)
#         self.conf = conf

#         # พารามิเตอร์ OCR ส่งต่อให้ plate_ocr.read_plate_array แบบเดิมทุกอย่าง
#         self.split_ratio = split_ratio
#         self.province_threshold = province_threshold
#         self.ocr_conf_threshold = ocr_conf_threshold

#     def detect(self, img_path):
#         img = cv2.imread(img_path)

#         if img is None:
#             raise ValueError("Image not found")

#         results = self.model(img)[0]

#         detections = []

#         for box in results.boxes:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])

#             # 🔥 filter class
#             if cls not in USE_CLASSES:
#                 continue

#             if conf < self.conf:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # กันหลุดภาพ
#             h, w = img.shape[:2]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)

#             crop = img[y1:y2, x1:x2]

#             det = {
#                 "class_id": cls,
#                 "bbox": (x1, y1, x2, y2),
#                 "conf": conf,
#                 "plate_number": None,
#                 "province": None,
#                 "province_score": 0,
#                 "valid_plate": False,
#             }

#             # OCR เฉพาะกล่องป้ายทะเบียน ส่ง crop ตรงเข้า plate_ocr โดยไม่เขียนไฟล์
#             if cls == PLATE_CLASS_ID and crop.size > 0:
#                 try:
#                     ocr_result = read_plate_array(
#                         crop,
#                         split_ratio=self.split_ratio,
#                         province_threshold=self.province_threshold,
#                         conf_threshold=self.ocr_conf_threshold,
#                         debug=False,
#                     )
#                     det["plate_number"]   = ocr_result["plate_number"]
#                     det["province"]       = ocr_result["province"]
#                     det["province_score"] = ocr_result["province_score"]
#                     det["valid_plate"]    = ocr_result["valid_plate"]
#                     det["ocr_raw"]        = ocr_result["province_raw"]
#                 except ValueError:
#                     pass

#             detections.append(det)

#         return img, detections

import cv2
from ultralytics import YOLO
from plate_ocr import read_plate_array

USE_CLASSES = [0, 1]   # 0 = car, 1 = license_plate
PLATE_CLASS_ID = 1     # class id ของป้ายทะเบียน (ใช้ค่านี้ที่เดียว ไม่ hardcode ซ้ำที่อื่น)


class PlateDetector:
    def __init__(self, model_path, conf=0.3,
                 split_ratio=0.60, province_threshold=70):
        self.model = YOLO(model_path)
        self.conf = conf

        # พารามิเตอร์ OCR ส่งต่อให้ plate_ocr.read_plate_array แบบเดิมทุกอย่าง
        self.split_ratio = split_ratio
        self.province_threshold = province_threshold


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

            det = {
                "class_id": cls,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "plate_number": None,
                "province": None,
                "province_score": 0,
                "valid_plate": False,
            }

            # OCR เฉพาะกล่องป้ายทะเบียน ส่ง crop ตรงเข้า plate_ocr โดยไม่เขียนไฟล์
            if cls == PLATE_CLASS_ID and crop.size > 0:
                try:
                    ocr_result = read_plate_array(
                        crop,
                        split_ratio=self.split_ratio,
                        province_threshold=self.province_threshold,
                        debug=False,
                    )
                    det["plate_number"]   = ocr_result["plate_number"]
                    det["province"]       = ocr_result["province"]
                    det["province_score"] = ocr_result["province_score"]
                    det["valid_plate"]    = ocr_result["valid_plate"]
                    det["ocr_raw"]        = ocr_result["province_raw"]
                except ValueError:
                    pass

            detections.append(det)

        return img, detections
