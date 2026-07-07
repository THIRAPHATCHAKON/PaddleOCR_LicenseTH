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
# Correction maps
# ─────────────────────────────────────────
NUM_MAP = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}

THAI_MAP = {
    "@": "ฮ",
    "&": "ฃ",
    "N": "ก",
    "n": "ก",
}

# ─────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────
PLATE_PATTERN = re.compile(r'^[0-9]?[ก-ฮ]{1,3}\s?[0-9]{1,4}$')

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
    for old, new in NUM_MAP.items():
        text = text.replace(old, new)
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
    h, w = image.shape[:2]
    cut = int(h * split_ratio)
    return image[:cut, :], image[cut:, :]


def ocr_image(image: np.ndarray, conf_threshold: float = 0.6):
    """อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก (ตามงานวิจัยโรมาเนีย)"""
    output = []
    for res in ocr.predict(image):
        text, score = parse_ocr_result(res)
        if score >= conf_threshold:
            output.append((text, score))
    return output


# ─────────────────────────────────────────
# Core: อ่านป้ายทะเบียน 1 ภาพ
# ─────────────────────────────────────────
def read_plate(image_path: str, split_ratio: float = 0.60,
               province_threshold: int = 70,
               conf_threshold: float = 0.6,
               debug: bool = True) -> dict:

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

    top_img, bottom_img = split_plate_image(img, split_ratio)

    # OCR พร้อม confidence threshold ตามงานวิจัยโรมาเนีย
    top_results      = ocr_image(top_img, conf_threshold)
    plate_number     = clean_plate_number(" ".join(r[0] for r in top_results))

    bottom_results       = ocr_image(bottom_img, conf_threshold)
    province_raw         = " ".join(r[0] for r in bottom_results)
    province_cleaned     = clean_province_text(province_raw)
    province, prov_score = match_province(province_cleaned, threshold=province_threshold)

    valid = bool(PLATE_PATTERN.match(plate_number))

    result = {
        "file":           Path(image_path).name,
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
                conf_threshold: float = 0.6,
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
                conf_threshold=conf_threshold,
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
    read_folder(folder, split_ratio=0.60, province_threshold=70, conf_threshold=0.6, debug=True)

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

# # ใช้แยกโซน: [เลขนำ]? [พยัญชนะไทย 1-3 ตัว] [เลขตาม]
# # อนุญาตให้ prefix/suffix มีอักษรละตินหรือ | ที่ยังไม่ถูกแปลง เผื่อไว้ก่อนแก้
# PLATE_SPLIT_PATTERN = re.compile(r'^([A-Za-z0-9|]*?)([ก-ฮ]{1,3})([A-Za-z0-9|]*)$')

# IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# DEBUG_OCR = True  # เปิด/ปิด debug raw OCR แบบละเอียด (ก่อน clean)

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


# def clean_plate_number(text: str, debug: bool = False) -> str:
#     """
#     แก้ไขข้อความป้ายทะเบียนแบบ 'รู้ตำแหน่ง' (position-aware)
#     เพื่อไม่ให้ NUM_MAP ไปทับพยัญชนะไทยที่อยู่ตรงกลาง เช่น
#     ตัว 'อ'/'ก' ที่โมเดลอ่านผิดเป็น 'O' -> ไม่ควรถูกแปลงเป็น '0'
#     ถ้ามันอยู่ในโซนที่ควรเป็นพยัญชนะ
#     """
#     raw = text
#     text = text.strip()

#     # 1. แปลงสัญลักษณ์ที่คล้ายอักษรไทยก่อน (เช่น @ -> ฮ)
#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)
#     after_thai_map = text

#     # 2. เก็บเฉพาะ อักษรไทย / ตัวเลข / ตัวละติน / '|' (ตัวละตินไว้เผื่อแก้ตำแหน่งในขั้นถัดไป)
#     text = re.sub(r'[^ก-๙0-9A-Za-z|]', '', text)
#     after_charset_filter = text

#     # 3. หาโซนพยัญชนะไทย แล้วแก้ NUM_MAP เฉพาะโซนตัวเลข (prefix/suffix) เท่านั้น
#     m = PLATE_SPLIT_PATTERN.match(text)
#     if m:
#         prefix, thai_part, suffix = m.groups()
#         for old, new in NUM_MAP.items():
#             prefix = prefix.replace(old, new)
#             suffix = suffix.replace(old, new)
#         text = prefix + thai_part + suffix
#         split_info = f"prefix={prefix!r} thai={thai_part!r} suffix={suffix!r}"
#     else:
#         # ไม่พบพยัญชนะไทยเลย -> เคสนี้น่าสงสัยว่าตัวอักษรไทยโดนอ่านผิดหมด
#         # ใช้ NUM_MAP แบบเดิม (blind replace) เป็น fallback
#         for old, new in NUM_MAP.items():
#             text = text.replace(old, new)
#         split_info = "ไม่พบพยัญชนะไทย -> ใช้ NUM_MAP แบบ fallback (blind replace)"

#     after_num_map = text

#     # 4. ลบอักษรที่ไม่ใช่ไทย/เลขทิ้งในขั้นสุดท้าย (เผื่อมีตัวละตินหลงเหลือจาก fallback)
#     text = re.sub(r'[^ก-๙0-9]', '', text)

#     if debug:
#         print("      [clean_plate_number]")
#         print(f"        raw input           : {raw!r}")
#         print(f"        after THAI_MAP       : {after_thai_map!r}")
#         print(f"        after charset filter  : {after_charset_filter!r}")
#         print(f"        split                : {split_info}")
#         print(f"        after NUM_MAP         : {after_num_map!r}")
#         print(f"        final                : {text!r}")

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


# def ocr_image(image: np.ndarray, conf_threshold: float = 0.6, tag: str = "", debug: bool = False):
#     """
#     อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก
#     ถ้า debug=True จะพิมพ์ผลลัพธ์ 'ทุกตัว' รวมถึงตัวที่ถูกกรองออก
#     เพื่อให้เห็นว่ามีข้อความไทยที่ conf ต่ำกว่าเกณฑ์ถูกทิ้งไปหรือไม่
#     """
#     output = []
#     all_raw = []
#     for res in ocr.predict(image):
#         text, score = parse_ocr_result(res)
#         all_raw.append((text, score))
#         if score >= conf_threshold:
#             output.append((text, score))

#     if debug:
#         print(f"      [ocr_image:{tag}] conf_threshold={conf_threshold}")
#         if not all_raw:
#             print("        (ไม่มีผลลัพธ์ OCR เลย)")
#         for text, score in all_raw:
#             kept = "✅ เก็บ" if score >= conf_threshold else "❌ ถูกกรองออก (conf ต่ำกว่าเกณฑ์)"
#             print(f"        raw={text!r:<20} conf={score:.4f}  -> {kept}")

#     return output


# # ─────────────────────────────────────────
# # Core: อ่านป้ายทะเบียน 1 ภาพ
# # ─────────────────────────────────────────
# def read_plate(image_path: str, split_ratio: float = 0.60,
#                province_threshold: int = 70,
#                conf_threshold: float = 0.6,
#                debug: bool = True) -> dict:

#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

#     if debug:
#         print(f"   [read_plate] ขนาดภาพเต็ม: {img.shape}")

#     top_img, bottom_img = split_plate_image(img, split_ratio)

#     if debug:
#         print(f"   [read_plate] split_ratio={split_ratio} -> top: {top_img.shape}, bottom: {bottom_img.shape}")

#     # OCR พร้อม confidence threshold ตามงานวิจัยโรมาเนีย
#     top_results = ocr_image(top_img, conf_threshold, tag="TOP(ทะเบียน)", debug=(debug and DEBUG_OCR))
#     top_joined  = " ".join(r[0] for r in top_results)
#     plate_number = clean_plate_number(top_joined, debug=(debug and DEBUG_OCR))

#     bottom_results   = ocr_image(bottom_img, conf_threshold, tag="BOTTOM(จังหวัด)", debug=(debug and DEBUG_OCR))
#     province_raw      = " ".join(r[0] for r in bottom_results)
#     province_cleaned   = clean_province_text(province_raw)
#     province, prov_score = match_province(province_cleaned, threshold=province_threshold)

#     valid = bool(PLATE_PATTERN.match(plate_number))

#     result = {
#         "file":           Path(image_path).name,
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
# # Debug เดี่ยว: ดู raw OCR + เซฟภาพที่ตัดไว้ดู โดยไม่ผ่านการกรอง conf เลย
# # ─────────────────────────────────────────
# def debug_raw_ocr(image_path: str, split_ratio: float = 0.60, save_crops: bool = True):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

#     top_img, bottom_img = split_plate_image(img, split_ratio)

#     print(f"\n{'#' * 60}")
#     print(f"# DEBUG RAW OCR: {image_path}")
#     print(f"{'#' * 60}")
#     print(f"ขนาดภาพเต็ม: {img.shape}, top: {top_img.shape}, bottom: {bottom_img.shape}")

#     if save_crops:
#         top_out = "debug_top.png"
#         bottom_out = "debug_bottom.png"
#         cv2.imwrite(top_out, top_img)
#         cv2.imwrite(bottom_out, bottom_img)
#         print(f"บันทึกภาพที่ตัดไว้ที่: {top_out}, {bottom_out}")

#     print("\n--- TOP (ไม่กรอง conf เลย) ---")
#     for res in ocr.predict(top_img):
#         text, score = parse_ocr_result(res)
#         print(f"  raw={text!r:<20} conf={score:.4f}")
#         cleaned = clean_plate_number(text, debug=True)
#         print(f"  -> plate_number cleaned: {cleaned!r}\n")

#     print("--- BOTTOM (ไม่กรอง conf เลย) ---")
#     for res in ocr.predict(bottom_img):
#         text, score = parse_ocr_result(res)
#         print(f"  raw={text!r:<20} conf={score:.4f}")


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

#     # ตัวอย่างการเรียก debug เจาะภาพเดี่ยว (ไม่กรอง conf เลย, เซฟ crop ให้ดูด้วยตา):
#     debug_raw_ocr(r"D:\license_plate_and_car_detection\test_img\img10.png")
#     debug_raw_ocr(r"D:\license_plate_and_car_detection\test_img\img11.png")
