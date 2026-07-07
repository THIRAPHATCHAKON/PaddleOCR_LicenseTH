# import re
# import sys
# import cv2
# import numpy as np
# from pathlib import Path
# from paddleocr import TextRecognition
# from rapidfuzz import process, fuzz

# # ─────────────────────────────────────────
# # OCR Model
# # ─────────────────────────────────────────
# ocr = TextRecognition(model_name="th_PP-OCRv5_mobile_rec")

# # ─────────────────────────────────────────
# # Correction maps
# # ─────────────────────────────────────────
# NUM_MAP = {
#     "O": "0", "Q": "0", "D": "0",
#     "I": "1", "l": "1", "|": "1",
#     "Z": "2",
#     "S": "5",
#     "B": "8",
# }

# THAI_MAP = {
#     "@": "ฮ",
#     "&": "ฃ",
#     "N": "ก",
#     "n": "ก",
# }

# # ─────────────────────────────────────────
# # Patterns
# # ─────────────────────────────────────────
# PLATE_PATTERN = re.compile(r'^[0-9]?[ก-ฮ]{1,3}\s?[0-9]{1,4}$')

# IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # ─────────────────────────────────────────
# # จังหวัด 77 จังหวัด
# # ─────────────────────────────────────────
# THAI_PROVINCES = [
#     "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
#     "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
#     "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
#     "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
#     "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
#     "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
#     "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
#     "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
#     "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
#     "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
#     "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
#     "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
#     "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
#     "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
#     "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
#     "อุทัยธานี", "อุบลราชธานี",
# ]


# # ─────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────
# def parse_ocr_result(res):
#     """รองรับทั้ง dict และ object แบบ PaddleOCR v3"""
#     if isinstance(res, dict):
#         data = res.get("res", res)
#         return data.get("rec_text", ""), data.get("rec_score", 0.0)
#     if hasattr(res, "rec_text"):
#         return res.rec_text, getattr(res, "rec_score", 0.0)
#     if hasattr(res, "__dict__"):
#         d = vars(res)
#         return d.get("rec_text", ""), d.get("rec_score", 0.0)
#     return str(res), 0.0


# def clean_plate_number(text: str) -> str:
#     text = text.strip()
#     for old, new in NUM_MAP.items():
#         text = text.replace(old, new)
#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)
#     text = re.sub(r'[^ก-๙0-9]', '', text)
#     return text


# def clean_province_text(text: str) -> str:
#     text = text.strip()
#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)
#     text = re.sub(r'[^ก-๙]', '', text)
#     return text


# def match_province(text: str, threshold: int = 70):
#     if not text:
#         return None, 0
#     result = process.extractOne(
#         text,
#         THAI_PROVINCES,
#         scorer=fuzz.token_set_ratio,
#     )
#     if result is None:
#         return None, 0
#     match, score, _ = result
#     if score >= threshold:
#         return match, score
#     return None, score


# def split_plate_image(image: np.ndarray, split_ratio: float = 0.60):
#     h, w = image.shape[:2]
#     cut = int(h * split_ratio)
#     return image[:cut, :], image[cut:, :]


# def ocr_image(image: np.ndarray, conf_threshold: float = 0.6):
#     """อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก (ตามงานวิจัยโรมาเนีย)"""
#     output = []
#     for res in ocr.predict(image):
#         text, score = parse_ocr_result(res)
#         if score >= conf_threshold:
#             output.append((text, score))
#     return output


# # ─────────────────────────────────────────
# # Core: อ่านป้ายทะเบียนจากภาพที่โหลดไว้แล้ว (np.ndarray)
# # ใช้วิธี OCR แบบเดิมทั้งหมด (split บน/ล่าง -> ocr -> clean -> match จังหวัด)
# # เพียงแต่ไม่ผูกกับไฟล์บนดิสก์ เพื่อให้ detect.py ส่ง crop ของ
# # กล่องป้ายทะเบียนเข้ามาตรง ๆ ได้
# # ─────────────────────────────────────────
# def read_plate_array(img: np.ndarray, split_ratio: float = 0.60,
#                      province_threshold: int = 70,
#                      conf_threshold: float = 0.6,
#                      debug: bool = False,
#                      file_name: str = "") -> dict:

#     if img is None or img.size == 0:
#         raise ValueError("ภาพที่ส่งเข้ามาว่างเปล่า")

#     top_img, bottom_img = split_plate_image(img, split_ratio)

#     # OCR พร้อม confidence threshold ตามงานวิจัยโรมาเนีย
#     top_results      = ocr_image(top_img, conf_threshold)
#     plate_number     = clean_plate_number(" ".join(r[0] for r in top_results))

#     bottom_results       = ocr_image(bottom_img, conf_threshold)
#     province_raw         = " ".join(r[0] for r in bottom_results)
#     province_cleaned     = clean_province_text(province_raw)
#     province, prov_score = match_province(province_cleaned, threshold=province_threshold)

#     valid = bool(PLATE_PATTERN.match(plate_number))

#     result = {
#         "file":           file_name,
#         "plate_number":   plate_number,
#         "province_raw":   province_raw,
#         "province":       province,
#         "province_score": prov_score,
#         "valid_plate":    valid,
#         "top_raw":        top_results,
#         "bottom_raw":     bottom_results,
#     }

#     if debug:
#         _print_result(result)

#     return result


# # ─────────────────────────────────────────
# # Core: อ่านป้ายทะเบียน 1 ภาพ (จาก path บนดิสก์)
# # เป็นแค่ wrapper รอบ read_plate_array — วิธี OCR เหมือนเดิมทุกอย่าง
# # ─────────────────────────────────────────
# def read_plate(image_path: str, split_ratio: float = 0.60,
#                province_threshold: int = 70,
#                conf_threshold: float = 0.6,
#                debug: bool = True) -> dict:

#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

#     return read_plate_array(
#         img,
#         split_ratio=split_ratio,
#         province_threshold=province_threshold,
#         conf_threshold=conf_threshold,
#         debug=debug,
#         file_name=Path(image_path).name,
#     )


# def _print_result(r: dict):
#     sep = "=" * 55
#     print(sep)
#     print(f"📁  {r['file']}")
#     print("📋  ส่วนบน (ทะเบียน)")
#     for text, score in r["top_raw"]:
#         print(f"   OCR Raw    : {text!r}  (conf={score:.3f})")
#     print(f"   → Cleaned  : {r['plate_number']}")
#     print(f"   → Valid    : {'✅ VALID' if r['valid_plate'] else '❌ INVALID'}")
#     print()
#     print("🗺️  ส่วนล่าง (จังหวัด)")
#     for text, score in r["bottom_raw"]:
#         print(f"   OCR Raw    : {text!r}  (conf={score:.3f})")
#     print(f"   → Cleaned  : {r['province_raw']}")
#     if r["province"]:
#         print(f"   → Province : {r['province']}  (fuzzy={r['province_score']})")
#     else:
#         print(f"   → Province : ไม่พบจังหวัด  (fuzzy={r['province_score']})")
#     print(sep)


# # ─────────────────────────────────────────
# # Batch: อ่านทุกภาพในโฟลเดอร์
# # ─────────────────────────────────────────
# def read_folder(folder_path: str, split_ratio: float = 0.60,
#                 province_threshold: int = 70,
#                 conf_threshold: float = 0.6,
#                 debug: bool = True) -> list:

#     folder = Path(folder_path)
#     if not folder.exists():
#         raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {folder_path}")

#     image_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)

#     if not image_files:
#         print(f"⚠️  ไม่พบไฟล์ภาพใน {folder_path}")
#         return []

#     print(f"🔍  พบ {len(image_files)} ภาพใน {folder_path}\n")

#     all_results = []
#     for i, img_path in enumerate(image_files, 1):
#         print(f"[{i}/{len(image_files)}] {img_path.name}")
#         try:
#             result = read_plate(
#                 str(img_path),
#                 split_ratio=split_ratio,
#                 province_threshold=province_threshold,
#                 conf_threshold=conf_threshold,
#                 debug=debug,
#             )
#             all_results.append(result)
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
#             all_results.append({
#                 "file": img_path.name,
#                 "plate_number": "", "province": None,
#                 "province_score": 0, "valid_plate": False,
#                 "error": str(e),
#             })

#     # ── ตารางสรุป ──
#     print("\n" + "=" * 65)
#     print(f"{'ไฟล์':<30} {'ทะเบียน':<14} {'จังหวัด':<16} {'Valid'}")
#     print("-" * 65)
#     for r in all_results:
#         plate = r.get("plate_number") or "ERROR"
#         prov  = r.get("province") or f"? ({r.get('province_score', 0)})"
#         valid = "✅" if r.get("valid_plate") else "❌"
#         print(f"{r['file']:<30} {plate:<14} {prov:<16} {valid}")
#     print("=" * 65)

#     return all_results


# # ─────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────
# if __name__ == "__main__":
#     folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\license_plate_and_car_detection\test_img"
#     read_folder(folder, split_ratio=0.60, province_threshold=70, conf_threshold=0.6, debug=True)

# import re
# import sys
# import cv2
# import numpy as np
# from pathlib import Path
# from paddleocr import TextRecognition
# from rapidfuzz import process, fuzz

# # ─────────────────────────────────────────
# # OCR Model
# # ─────────────────────────────────────────
# ocr = TextRecognition(model_name="th_PP-OCRv5_mobile_rec")

# # ─────────────────────────────────────────
# # Correction maps
# # ─────────────────────────────────────────
# NUM_MAP = {
#     "O": "0", "Q": "0", "D": "0",
#     "I": "1", "l": "1", "|": "1",
#     "Z": "2",
#     "S": "5",
#     "B": "8",
# }

# THAI_MAP = {
#     "@": "ฮ",
#     "&": "ฃ",
#     "N": "ก",
#     "n": "ก",
# }

# # ─────────────────────────────────────────
# # Patterns
# # ป้ายทะเบียนรถยนต์ส่วนบุคคลทั่วไป (ป้ายขาว ไม่เกิน 7 ที่นั่ง) รวมป้ายประมูล:
# #   [เลขนำหน้า 1 หลัก 1-9]?  ตัวอักษรไทย 1-2 ตัว  [เว้นวรรค]?  ตัวเลข 1-4 หลัก (ห้ามนำด้วย 0)
# #
# # ตัวอย่างที่ผ่าน:  ก1234 / กข5678 / 2กข1234 / กก1 / กก88
# # ตัวอย่างที่ไม่ผ่าน: 3 ตัวอักษร (กขค1234), เลขนำ 0 (0012, กข0123), เลขเกิน 4 หลัก
# # ─────────────────────────────────────────
# PLATE_PATTERN = re.compile(r'^[1-9]?[ก-ฮ]{1,2}\s?[1-9][0-9]{0,3}$')

# IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # ─────────────────────────────────────────
# # จังหวัด 77 จังหวัด
# # ─────────────────────────────────────────
# THAI_PROVINCES = [
#     "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
#     "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
#     "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
#     "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
#     "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
#     "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
#     "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
#     "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
#     "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
#     "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
#     "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
#     "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
#     "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
#     "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
#     "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
#     "อุทัยธานี", "อุบลราชธานี",
# ]


# # ─────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────
# def parse_ocr_result(res):
#     """รองรับทั้ง dict และ object แบบ PaddleOCR v3"""
#     if isinstance(res, dict):
#         data = res.get("res", res)
#         return data.get("rec_text", ""), data.get("rec_score", 0.0)
#     if hasattr(res, "rec_text"):
#         return res.rec_text, getattr(res, "rec_score", 0.0)
#     if hasattr(res, "__dict__"):
#         d = vars(res)
#         return d.get("rec_text", ""), d.get("rec_score", 0.0)
#     return str(res), 0.0


# def clean_plate_number(text: str) -> str:
#     text = text.strip()
#     for old, new in NUM_MAP.items():
#         text = text.replace(old, new)
#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)
#     text = re.sub(r'[^ก-๙0-9]', '', text)
#     return text


# def clean_province_text(text: str) -> str:
#     text = text.strip()
#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)
#     text = re.sub(r'[^ก-๙]', '', text)
#     return text


# def match_province(text: str, threshold: int = 70):
#     if not text:
#         return None, 0
#     result = process.extractOne(
#         text,
#         THAI_PROVINCES,
#         scorer=fuzz.token_set_ratio,
#     )
#     if result is None:
#         return None, 0
#     match, score, _ = result
#     if score >= threshold:
#         return match, score
#     return None, score


# def split_plate_image(image: np.ndarray, split_ratio: float = 0.60):
#     h, w = image.shape[:2]
#     cut = int(h * split_ratio)
#     return image[:cut, :], image[cut:, :]


# def ocr_image(image: np.ndarray, conf_threshold: float = 0.6):
#     """อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก (ตามงานวิจัยโรมาเนีย)"""
#     output = []
#     for res in ocr.predict(image):
#         text, score = parse_ocr_result(res)
#         if score >= conf_threshold:
#             output.append((text, score))
#     return output


# # ─────────────────────────────────────────
# # Core: อ่านป้ายทะเบียนจากภาพที่โหลดไว้แล้ว (np.ndarray)
# # ใช้วิธี OCR แบบเดิมทั้งหมด (split บน/ล่าง -> ocr -> clean -> match จังหวัด)
# # เพียงแต่ไม่ผูกกับไฟล์บนดิสก์ เพื่อให้ detect.py ส่ง crop ของ
# # กล่องป้ายทะเบียนเข้ามาตรง ๆ ได้
# # ─────────────────────────────────────────
# def read_plate_array(img: np.ndarray, split_ratio: float = 0.60,
#                      province_threshold: int = 70,
#                      conf_threshold: float = 0.6,
#                      debug: bool = False,
#                      file_name: str = "") -> dict:

#     if img is None or img.size == 0:
#         raise ValueError("ภาพที่ส่งเข้ามาว่างเปล่า")

#     top_img, bottom_img = split_plate_image(img, split_ratio)

#     # OCR พร้อม confidence threshold ตามงานวิจัยโรมาเนีย
#     top_results      = ocr_image(top_img, conf_threshold)
#     plate_number     = clean_plate_number(" ".join(r[0] for r in top_results))

#     bottom_results       = ocr_image(bottom_img, conf_threshold)
#     province_raw         = " ".join(r[0] for r in bottom_results)
#     province_cleaned     = clean_province_text(province_raw)
#     province, prov_score = match_province(province_cleaned, threshold=province_threshold)

#     valid = bool(PLATE_PATTERN.match(plate_number))

#     result = {
#         "file":           file_name,
#         "plate_number":   plate_number,
#         "province_raw":   province_raw,
#         "province":       province,
#         "province_score": prov_score,
#         "valid_plate":    valid,
#         "top_raw":        top_results,
#         "bottom_raw":     bottom_results,
#     }

#     if debug:
#         _print_result(result)

#     return result


# # ─────────────────────────────────────────
# # Core: อ่านป้ายทะเบียน 1 ภาพ (จาก path บนดิสก์)
# # เป็นแค่ wrapper รอบ read_plate_array — วิธี OCR เหมือนเดิมทุกอย่าง
# # ─────────────────────────────────────────
# def read_plate(image_path: str, split_ratio: float = 0.60,
#                province_threshold: int = 70,
#                conf_threshold: float = 0.6,
#                debug: bool = True) -> dict:

#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

#     return read_plate_array(
#         img,
#         split_ratio=split_ratio,
#         province_threshold=province_threshold,
#         conf_threshold=conf_threshold,
#         debug=debug,
#         file_name=Path(image_path).name,
#     )


# def _print_result(r: dict):
#     sep = "=" * 55
#     print(sep)
#     print(f"📁  {r['file']}")
#     print("📋  ส่วนบน (ทะเบียน)")
#     for text, score in r["top_raw"]:
#         print(f"   OCR Raw    : {text!r}  (conf={score:.3f})")
#     print(f"   → Cleaned  : {r['plate_number']}")
#     print(f"   → Valid    : {'✅ VALID' if r['valid_plate'] else '❌ INVALID'}")
#     print()
#     print("🗺️  ส่วนล่าง (จังหวัด)")
#     for text, score in r["bottom_raw"]:
#         print(f"   OCR Raw    : {text!r}  (conf={score:.3f})")
#     print(f"   → Cleaned  : {r['province_raw']}")
#     if r["province"]:
#         print(f"   → Province : {r['province']}  (fuzzy={r['province_score']})")
#     else:
#         print(f"   → Province : ไม่พบจังหวัด  (fuzzy={r['province_score']})")
#     print(sep)


# # ─────────────────────────────────────────
# # Batch: อ่านทุกภาพในโฟลเดอร์
# # ─────────────────────────────────────────
# def read_folder(folder_path: str, split_ratio: float = 0.60,
#                 province_threshold: int = 70,
#                 conf_threshold: float = 0.6,
#                 debug: bool = True) -> list:

#     folder = Path(folder_path)
#     if not folder.exists():
#         raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {folder_path}")

#     image_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)

#     if not image_files:
#         print(f"⚠️  ไม่พบไฟล์ภาพใน {folder_path}")
#         return []

#     print(f"🔍  พบ {len(image_files)} ภาพใน {folder_path}\n")

#     all_results = []
#     for i, img_path in enumerate(image_files, 1):
#         print(f"[{i}/{len(image_files)}] {img_path.name}")
#         try:
#             result = read_plate(
#                 str(img_path),
#                 split_ratio=split_ratio,
#                 province_threshold=province_threshold,
#                 conf_threshold=conf_threshold,
#                 debug=debug,
#             )
#             all_results.append(result)
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
#             all_results.append({
#                 "file": img_path.name,
#                 "plate_number": "", "province": None,
#                 "province_score": 0, "valid_plate": False,
#                 "error": str(e),
#             })

#     # ── ตารางสรุป ──
#     print("\n" + "=" * 65)
#     print(f"{'ไฟล์':<30} {'ทะเบียน':<14} {'จังหวัด':<16} {'Valid'}")
#     print("-" * 65)
#     for r in all_results:
#         plate = r.get("plate_number") or "ERROR"
#         prov  = r.get("province") or f"? ({r.get('province_score', 0)})"
#         valid = "✅" if r.get("valid_plate") else "❌"
#         print(f"{r['file']:<30} {plate:<14} {prov:<16} {valid}")
#     print("=" * 65)

#     return all_results


# # ─────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────
# if __name__ == "__main__":
#     folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\license_plate_and_car_detection\test_img"
#     read_folder(folder, split_ratio=0.60, province_threshold=70, conf_threshold=0.6, debug=True)

import re
import sys
import cv2
import numpy as np
from pathlib import Path
from paddleocr import TextRecognition
from rapidfuzz import process, fuzz

# ─────────────────────────────────────────
# OCR Model
# ─────────────────────────────────────────
ocr = TextRecognition(model_name="th_PP-OCRv5_mobile_rec")

# ─────────────────────────────────────────
# Correction map (แก้ symbol ที่ OCR เดาผิดให้เป็นตัวอักษรไทยที่ถูกต้อง)
# ─────────────────────────────────────────
THAI_MAP = {
    "@": "ฮ",
    "&": "ฃ",
    "N": "ก",
    "n": "ก",
}

# ─────────────────────────────────────────
# Patterns
# ป้ายทะเบียนรถยนต์ส่วนบุคคลทั่วไป (ป้ายขาว ไม่เกิน 7 ที่นั่ง) รวมป้ายประมูล:
#   [เลขนำหน้า 1 หลัก 1-9]?  ตัวอักษรไทย 1-2 ตัว  [เว้นวรรค]?  ตัวเลข 1-4 หลัก (ห้ามนำด้วย 0)
#
# ตัวอย่างที่ผ่าน:  ก1234 / กข5678 / 2กข1234 / กก1 / กก88
# ตัวอย่างที่ไม่ผ่าน: 3 ตัวอักษร (กขค1234), เลขนำ 0 (0012, กข0123), เลขเกิน 4 หลัก
# ─────────────────────────────────────────
PLATE_PATTERN = re.compile(r'^[1-9]?[ก-ฮ]{1,2}\s?[1-9][0-9]{0,3}$')

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────
# จังหวัด 77 จังหวัด
# ─────────────────────────────────────────
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี",
]


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def parse_ocr_result(res):
    """รองรับทั้ง dict และ object แบบ PaddleOCR v3"""
    if isinstance(res, dict):
        data = res.get("res", res)
        return data.get("rec_text", ""), data.get("rec_score", 0.0)
    if hasattr(res, "rec_text"):
        return res.rec_text, getattr(res, "rec_score", 0.0)
    if hasattr(res, "__dict__"):
        d = vars(res)
        return d.get("rec_text", ""), d.get("rec_score", 0.0)
    return str(res), 0.0


def clean_plate_number(text: str) -> str:
    text = text.strip()
    for old, new in THAI_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r'[^ก-๙0-9]', '', text)
    return text


def clean_province_text(text: str) -> str:
    text = text.strip()
    for old, new in THAI_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r'[^ก-๙]', '', text)
    return text


def match_province(text: str, threshold: int = 70):
    """ทายจังหวัดด้วย fuzzy matching (rapidfuzz)"""
    if not text:
        return None, 0
    result = process.extractOne(
        text,
        THAI_PROVINCES,
        scorer=fuzz.token_set_ratio,
    )
    if result is None:
        return None, 0
    match, score, _ = result
    if score >= threshold:
        return match, score
    return None, score


def split_plate_image(image: np.ndarray, split_ratio: float = 0.60):
    """แบ่งภาพป้ายเป็นบน (ทะเบียน) / ล่าง (จังหวัด) ตามสัดส่วน"""
    h, w = image.shape[:2]
    cut = int(h * split_ratio)
    return image[:cut, :], image[cut:, :]


def ocr_image(image: np.ndarray):
    """อ่าน OCR แล้วคืนผลลัพธ์ทั้งหมดที่ได้ (text, score)"""
    output = []
    for res in ocr.predict(image):
        text, score = parse_ocr_result(res)
        output.append((text, score))
    return output


# ─────────────────────────────────────────
# Core: อ่านป้ายทะเบียนจากภาพที่โหลดไว้แล้ว (np.ndarray)
# split บน/ล่าง -> ocr -> clean -> match จังหวัดด้วย fuzzy
# ไม่ผูกกับไฟล์บนดิสก์ เพื่อให้ detect.py ส่ง crop ของกล่องป้ายทะเบียน
# เข้ามาตรง ๆ ได้
# ─────────────────────────────────────────
def read_plate_array(img: np.ndarray, split_ratio: float = 0.60,
                     province_threshold: int = 70,
                     debug: bool = False,
                     file_name: str = "") -> dict:

    if img is None or img.size == 0:
        raise ValueError("ภาพที่ส่งเข้ามาว่างเปล่า")

    top_img, bottom_img = split_plate_image(img, split_ratio)

    top_results      = ocr_image(top_img)
    plate_number     = clean_plate_number(" ".join(r[0] for r in top_results))

    bottom_results       = ocr_image(bottom_img)
    province_raw         = " ".join(r[0] for r in bottom_results)
    province_cleaned     = clean_province_text(province_raw)
    province, prov_score = match_province(province_cleaned, threshold=province_threshold)

    valid = bool(PLATE_PATTERN.match(plate_number))

    result = {
        "file":           file_name,
        "plate_number":   plate_number,
        "province_raw":   province_raw,
        "province":       province,
        "province_score": prov_score,
        "valid_plate":    valid,
        "top_raw":        top_results,
        "bottom_raw":     bottom_results,
    }

    if debug:
        _print_result(result)

    return result


# ─────────────────────────────────────────
# Core: อ่านป้ายทะเบียน 1 ภาพ (จาก path บนดิสก์)
# เป็นแค่ wrapper รอบ read_plate_array
# ─────────────────────────────────────────
def read_plate(image_path: str, split_ratio: float = 0.60,
               province_threshold: int = 70,
               debug: bool = True) -> dict:

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

    return read_plate_array(
        img,
        split_ratio=split_ratio,
        province_threshold=province_threshold,
        debug=debug,
        file_name=Path(image_path).name,
    )


def _print_result(r: dict):
    sep = "=" * 55
    print(sep)
    print(f"📁  {r['file']}")
    print("📋  ส่วนบน (ทะเบียน)")
    for text, score in r["top_raw"]:
        print(f"   OCR Raw    : {text!r}  (conf={score:.3f})")
    print(f"   → Cleaned  : {r['plate_number']}")
    print(f"   → Valid    : {'✅ VALID' if r['valid_plate'] else '❌ INVALID'}")
    print()
    print("🗺️  ส่วนล่าง (จังหวัด)")
    for text, score in r["bottom_raw"]:
        print(f"   OCR Raw    : {text!r}  (conf={score:.3f})")
    print(f"   → Cleaned  : {r['province_raw']}")
    if r["province"]:
        print(f"   → Province : {r['province']}  (fuzzy={r['province_score']})")
    else:
        print(f"   → Province : ไม่พบจังหวัด  (fuzzy={r['province_score']})")
    print(sep)


# ─────────────────────────────────────────
# Batch: อ่านทุกภาพในโฟลเดอร์
# ─────────────────────────────────────────
def read_folder(folder_path: str, split_ratio: float = 0.60,
                province_threshold: int = 70,
                debug: bool = True) -> list:

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {folder_path}")

    image_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)

    if not image_files:
        print(f"⚠️  ไม่พบไฟล์ภาพใน {folder_path}")
        return []

    print(f"🔍  พบ {len(image_files)} ภาพใน {folder_path}\n")

    all_results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}")
        try:
            result = read_plate(
                str(img_path),
                split_ratio=split_ratio,
                province_threshold=province_threshold,
                debug=debug,
            )
            all_results.append(result)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results.append({
                "file": img_path.name,
                "plate_number": "", "province": None,
                "province_score": 0, "valid_plate": False,
                "error": str(e),
            })

    # ── ตารางสรุป ──
    print("\n" + "=" * 65)
    print(f"{'ไฟล์':<30} {'ทะเบียน':<14} {'จังหวัด':<16} {'Valid'}")
    print("-" * 65)
    for r in all_results:
        plate = r.get("plate_number") or "ERROR"
        prov  = r.get("province") or f"? ({r.get('province_score', 0)})"
        valid = "✅" if r.get("valid_plate") else "❌"
        print(f"{r['file']:<30} {plate:<14} {prov:<16} {valid}")
    print("=" * 65)

    return all_results


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\license_plate_and_car_detection\test_img"
    read_folder(folder, split_ratio=0.60, province_threshold=70, debug=True)