import re
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from paddleocr import TextRecognition
from ultralytics import YOLO
from rapidfuzz import process, fuzz
from collections import Counter
from itertools import combinations
import csv

# ============================================================
# CONFIG
# ============================================================

YOLO2_MODEL_PATH = (
    r"D:\license_plate_and_car_detection\model_yolo\license_version_4_320\best.pt"
)

YOLO2_IMG_SIZE = 320
YOLO2_CONF = 0.50



# crop จาก yolo2 (number/province) มักสูงแค่ ~20-40px — ถ้าปล่อยให้ ocr.predict()
# resize เข้า input shape ของมันเองเฉยๆ จะได้แค่ยืดภาพเบลอ ไม่ได้รายละเอียดเพิ่ม
# ดันความสูงขั้นต่ำตรงนี้ก่อนด้วย interpolation คุณภาพสูง คล้ายกับที่ yolo2 คุม
# ความคมชัด/ขนาดอินพุตด้วย YOLO2_IMG_SIZE
OCR_MIN_HEIGHT = 64
OCR_UPSCALE_INTERPOLATION = cv2.INTER_CUBIC

IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# IMAGE ENHANCEMENT (Adaptive Gamma + CLAHE)
# ============================================================
# ปิดไว้เป็นค่าเริ่มต้น (ดูเหตุผลใน docstring ของ enhance_image) — เปิดด้วย
# DETECT_ENHANCE=1 แล้ววัดผลจริงก่อนตัดสินใจย้ายมาเป็นค่าเริ่มต้น
DETECT_ENHANCE = os.environ.get("DETECT_ENHANCE", "0") == "1"

TARGET_BRIGHTNESS = 130      # ค่าความสว่างเฉลี่ย (0-255) ที่ adaptive_gamma ดันเข้าหา
GAMMA_RANGE = (0.4, 2.5)     # ขอบเขต gamma กันภาพมืด/สว่างจัดจน gamma เพี้ยนสุดขั้ว
CLAHE_CLIP = 2.0             # clipLimit ของ CLAHE บน L-channel
CLAHE_GRID = (8, 8)          # tileGridSize ของ CLAHE


# ============================================================
# LOAD MODEL
# ============================================================

yolo2 = YOLO(
    str(YOLO2_MODEL_PATH),
    task="detect"
)

ocr = TextRecognition(
    model_name="th_PP-OCRv5_mobile_rec",
    model_dir=r"D:\license_plate_and_car_detection\th_plate_rec_1"
)


# ============================================================
# YOLO CLASS
#
# names: ['number', 'province']
# ============================================================

NUMBER_CLASS_NAME = "number"
PROVINCE_CLASS_NAME = "province"


print("YOLO #2 Classes:")
print(yolo2.names)


# ============================================================
# WARM-UP
# ============================================================

_dummy_yolo = np.zeros(
    (
        YOLO2_IMG_SIZE,
        YOLO2_IMG_SIZE,
        3
    ),
    dtype=np.uint8
)

yolo2.predict(
    _dummy_yolo,
    imgsz=YOLO2_IMG_SIZE,
    conf=YOLO2_CONF,
    verbose=False
)


_dummy_ocr = np.zeros(
    (
        48,
        320,
        3
    ),
    dtype=np.uint8
)

for _ in ocr.predict(_dummy_ocr):
    pass


# ============================================================
# ADAPTIVE GAMMA + CLAHE ENHANCEMENT
# ============================================================

def adaptive_gamma(image):
    """ดันความสว่างเฉลี่ยเข้าหา TARGET_BRIGHTNESS — ภาพมืดได้ gamma < 1 ภาพโอเวอร์ได้ > 1"""
    mean = max(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0, 0.01)
    gamma = np.clip(np.log(TARGET_BRIGHTNESS / 255.0) / np.log(mean), *GAMMA_RANGE)

    table = (((np.arange(256) / 255.0) ** gamma) * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def enhance_image(image):
    """Adaptive Gamma + CLAHE บน L-channel

    ponytail: ปิดไว้เป็นค่าเริ่มต้นเพราะ best.onnx เทรนด้วยภาพดิบ — ป้อนภาพที่ปรับแล้ว
    คือเปลี่ยน distribution ที่โมเดลไม่เคยเห็น (domain shift) detection อาจตกแทนที่จะขึ้น
    และ CLAHE บน LAB ทั้งเฟรมกิน CPU ทุกเฟรม ซึ่งงบไม่เหลือ (ฝั่งอ่าน ~0.56 วิ/รูป ฝั่งถ่าย
    ป้อน 0.5 วิ/รูป) เปิดด้วย DETECT_ENHANCE=1 แล้ววัดด้วย --batch ว่าอ่านถูกขึ้นจริงไหม
    ก่อนตัดสินใจย้ายมาเป็นค่าเริ่มต้น
    """
    lab = cv2.cvtColor(adaptive_gamma(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ============================================================
# CONFUSION MAP
# ============================================================

# อัพเดทจาก ocr_new.py: แยก THAI_MAP (อังกฤษ/ตัวเลข -> ไทย ใช้ได้ทั้งโซนเลขทะเบียน
# และ clean_province_text) ออกจาก PLATE_ONLY_MAP (ไทย -> ไทย ใช้เฉพาะเลขทะเบียนเท่านั้น)
# เดิม "า":"ว" และ "ฤ":"ฎ" อยู่ปนใน THAI_MAP ตัวเดียว ทำให้ clean_province_text แทนที่
# "า" ในชื่อจังหวัดไปด้วย เช่น ตาก -> ตวก คะแนน fuzzy ตกจนหลุด threshold
THAI_MAP = {
    "@": "ฮ",
    "&": "ฃ",
    "N": "ก",
    "n": "ก",
    "1": "ก",
    "0": "ค",
    "H": "ฬ",
    "W": "พ",
    "U": "ข",
    "A": "ผ",
    "V": "ง",
    "v": "ง",
    "C": "ฌ",
}

# ไทย -> ไทย ใช้เฉพาะซ่อมโซนตัวอักษรของเลขทะเบียน ห้ามใช้กับ clean_province_text
PLATE_ONLY_MAP = {"า": "ว", "ฤ": "ฎ"}


DIGIT_MAP = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",

    "I": "1",
    "l": "1",
    "i": "1",

    "Z": "2",

    "E": "3",

    "A": "4",

    "S": "5",
    "s": "5",

    "G": "6",
    "b": "6",

    "T": "7",

    "B": "8",

    "g": "9",
    "q": "9",

    "m": "1",
}


# ============================================================
# VALID THAI CONSONANTS
# ============================================================

VALID_THAI_CONSONANTS = set(
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
)


# ============================================================
# PLATE KEEP WHITELIST (อัพเดทจาก ocr_new.py)
# ============================================================
# ตัวที่ยอมให้ผ่านเข้าไปซ่อมป้าย: เลข พยัญชนะไทยทั้งช่วง (ก-ฮ) ตัวอักษรอังกฤษ
# และทุกตัวที่มี map แปลงรออยู่ ที่เหลือคือขยะจากภาพ (กรอบป้าย/น็อต [ ] |,
# วรรณยุกต์ที่ป้ายไม่มี, จุด/ขีด/เว้นวรรคที่ rec อ่านติดมาจากขอบ crop) ทิ้งได้ปลอดภัย
import string as _string

_PLATE_KEEP = (
    set(_string.digits)
    | {chr(c) for c in range(0x0E01, 0x0E2F)}  # ก-ฮ ทั้งช่วง
    | set(_string.ascii_letters)
    | set(DIGIT_MAP)
    | set(THAI_MAP)
    | set(PLATE_ONLY_MAP)
)


# ============================================================
# PLATE PATTERN
# ============================================================

PLATE_PATTERN = re.compile(
    r'^[0-9]{0,1}[\u0E01-\u0E4E]{1,10}[0-9]{1,4}$' # r'^[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}$'
)

# อัพเดทจาก ocr_new.py: น้ำหนักที่เหลือของสตริงที่ผิดฟอร์แมตตอนโหวต (plate_vote)
BAD_PATTERN_WEIGHT = 0.4


# ============================================================
# THAI PROVINCES
# ============================================================

THAI_PROVINCES = [
    "กรุงเทพมหานคร",
    "กระบี่",
    "กาญจนบุรี",
    "กาฬสินธุ์",
    "กำแพงเพชร",
    "ขอนแก่น",
    "จันทบุรี",
    "ฉะเชิงเทรา",
    "ชลบุรี",
    "ชัยนาท",
    "ชัยภูมิ",
    "ชุมพร",
    "เชียงราย",
    "เชียงใหม่",
    "ตรัง",
    "ตราด",
    "ตาก",
    "นครนายก",
    "นครปฐม",
    "นครพนม",
    "นครราชสีมา",
    "นครศรีธรรมราช",
    "นครสวรรค์",
    "นนทบุรี",
    "นราธิวาส",
    "น่าน",
    "บึงกาฬ",
    "บุรีรัมย์",
    "ปทุมธานี",
    "ประจวบคีรีขันธ์",
    "ปราจีนบุรี",
    "ปัตตานี",
    "พระนครศรีอยุธยา",
    "พะเยา",
    "พังงา",
    "พัทลุง",
    "พิจิตร",
    "พิษณุโลก",
    "เพชรบุรี",
    "เพชรบูรณ์",
    "แพร่",
    "ภูเก็ต",
    "มหาสารคาม",
    "มุกดาหาร",
    "แม่ฮ่องสอน",
    "ยโสธร",
    "ยะลา",
    "ร้อยเอ็ด",
    "ระนอง",
    "ระยอง",
    "ราชบุรี",
    "ลพบุรี",
    "ลำปาง",
    "ลำพูน",
    "เลย",
    "ศรีสะเกษ",
    "สกลนคร",
    "สงขลา",
    "สตูล",
    "สมุทรปราการ",
    "สมุทรสงคราม",
    "สมุทรสาคร",
    "สระแก้ว",
    "สระบุรี",
    "สิงห์บุรี",
    "สุโขทัย",
    "สุพรรณบุรี",
    "สุราษฎร์ธานี",
    "สุรินทร์",
    "หนองคาย",
    "หนองบัวลำภู",
    "อ่างทอง",
    "อำนาจเจริญ",
    "อุดรธานี",
    "อุตรดิตถ์",
    "อุทัยธานี",
    "อุบลราชธานี",
    "เบตง",
]


# ============================================================
# YOLO #2 DETECTION
# ============================================================

def detect_plate_parts(
    plate_image,
    conf=YOLO2_CONF
):
    """
    YOLO #2

    Input:
        ภาพป้ายทะเบียน

    Output:
        number
        province

    Model classes:
        0 = number
        1 = province
    """

    if plate_image is None:
        raise ValueError(
            "plate_image is None"
        )

    if plate_image.size == 0:
        return [], []


    results = yolo2.predict(
        plate_image,
        conf=conf,
        imgsz=YOLO2_IMG_SIZE,
        verbose=False
    )


    plate_numbers = []
    provinces = []


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            cls = int(
                box.cls[0]
            )

            class_name = yolo2.names[cls]

            score = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # -----------------------------------------------
            # จำกัด Bounding Box
            # -----------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                plate_image.shape[1],
                x2
            )

            y2 = min(
                plate_image.shape[0],
                y2
            )


            if x2 <= x1 or y2 <= y1:
                continue


            crop = plate_image[
                y1:y2,
                x1:x2
            ].copy()


            detection = {
                "image": crop,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "conf": score,
                "class": class_name
            }


            # ===============================================
            # IMPORTANT
            #
            # YOLO MODEL:
            #
            # ['number', 'province']
            # ===============================================

            if class_name == NUMBER_CLASS_NAME:

                plate_numbers.append(
                    detection
                )


            elif class_name == PROVINCE_CLASS_NAME:

                provinces.append(
                    detection
                )


    # เรียงจากซ้ายไปขวา
    plate_numbers.sort(
        key=lambda d: d["bbox"][0]
    )

    provinces.sort(
        key=lambda d: d["bbox"][0]
    )


    return (
        plate_numbers,
        provinces
    )


# ============================================================
# OCR RESULT PARSER
# ============================================================

def parse_ocr_result(res):

    if isinstance(res, dict):

        data = res.get(
            "res",
            res
        )

        return (
            data.get(
                "rec_text",
                ""
            ),
            data.get(
                "rec_score",
                0.0
            )
        )


    if hasattr(
        res,
        "rec_text"
    ):

        return (
            res.rec_text,
            getattr(
                res,
                "rec_score",
                0.0
            )
        )


    if hasattr(
        res,
        "__dict__"
    ):

        d = vars(res)

        return (
            d.get(
                "rec_text",
                ""
            ),
            d.get(
                "rec_score",
                0.0
            )
        )


    return (
        str(res),
        0.0
    )


# ============================================================
# ขยาย CROP ก่อนเข้า OCR (คล้าย imgsz ของ yolo2)
# ============================================================

def prepare_ocr_crop(
    image,
    min_height=OCR_MIN_HEIGHT
):
    """ดันความสูงของ crop ให้ไม่ต่ำกว่า min_height px ก่อนป้อนเข้า ocr.predict()

    crop จาก detect_plate_parts() คือ pixel ดิบตรงๆ จากภาพต้นฉบับ ไม่ผ่านการ
    ควบคุมขนาด/ความคมใดๆ (ต่างจาก yolo2.predict ที่คุม imgsz=YOLO2_IMG_SIZE ชัดเจน)
    ถ้าปล่อยให้ rec model resize เข้า input shape ของมันเองตอน predict()
    จะได้แค่ยืดภาพเบลอเพิ่ม ไม่ได้รายละเอียดจริง — ขยายด้วย cv2.resize
    (INTER_CUBIC) ตรงนี้ก่อนแทน เพื่อให้เส้นตัวอักษรคมขึ้นก่อนเข้าโมเดล
    """

    if image is None or image.size == 0:
        return image


    h, w = image.shape[:2]

    if h >= min_height:
        return image


    scale = min_height / h

    new_w = max(
        1,
        int(round(w * scale))
    )

    return cv2.resize(
        image,
        (new_w, min_height),
        interpolation=OCR_UPSCALE_INTERPOLATION
    )


# ============================================================
# OCR IMAGE
# ============================================================

def ocr_image(
    image,
    conf_threshold=0.6
):

    if image is None:
        return []


    if image.size == 0:
        return []


    # ขยาย crop เล็กๆ จาก yolo2 ให้ใหญ่/คมขึ้นก่อนเข้า rec
    image = prepare_ocr_crop(
        image
    )


    output = []


    for res in ocr.predict(image):

        text, score = parse_ocr_result(
            res
        )


        if score >= conf_threshold:

            output.append(
                (
                    text,
                    score
                )
            )


    return output


# ============================================================
# CLEAN PLATE NUMBER
# ============================================================

def clean_plate_number(text):

    # กรองขยะจากภาพทิ้งก่อนซ่อม (กรอบป้าย/น็อต วรรณยุกต์ที่ไม่มีบนป้าย ฯลฯ)
    # เว้นวรรคหลุดออกไปพร้อมกันตรงนี้ ไม่ต้อง replace(" ", "") แยกอีกที
    raw = "".join(c for c in text if c in _PLATE_KEEP)

    if not raw:
        return ""


    fixed = _try_fix_plate(
        raw
    )


    if fixed:

        return fixed


    # ถ้า OCR อ่านได้แต่รูปแบบไม่ผ่าน
    # คืนค่า raw ไว้เพื่อ debug
    return raw


# ============================================================
# FIX PLATE
# ============================================================

def _try_fix_plate(raw):

    number_zone, letter_zone = (
        _split_number_letter_zone(
            raw
        )
    )


    if number_zone is None:
        return None


    fixed_number = _repair_digits(
        number_zone
    )


    if (
        not fixed_number
        or not fixed_number.isdigit()
    ):
        return None


    prefix = ""

    letters = letter_zone


    # --------------------------------------------------------
    # เลขนำหน้า เช่น 1กข1234
    # --------------------------------------------------------

    if letters:

        first = letters[0]


        if (
            first.isdigit()
            and first != "0"
        ):

            prefix = first
            letters = letters[1:]


        elif (
            first in DIGIT_MAP
            and DIGIT_MAP[first] != "0"
        ):

            prefix = DIGIT_MAP[first]
            letters = letters[1:]


    # --------------------------------------------------------
    # แก้ตัวอักษร
    # --------------------------------------------------------

    fixed_letters, dropped = _repair_letters(
        letters
    )

    # อัพเดทจาก ocr_new.py: ถ้ามีตัวที่แปลไม่ออกโดนทิ้งไป ห้ามปล่อยผ่านเงียบๆ
    # เพราะจะได้ป้ายสั้นลง 1 ตัวที่ "ตรงฟอร์แมต" แล้วกินน้ำหนักเต็มตอนโหวต
    # (1NX1234 -> 1ก1234) = เปลี่ยนผิดแบบเห็นได้ เป็นผิดแบบมั่นใจ ปล่อยให้คืน raw แทน
    if dropped:
        return None


    if len(fixed_letters) > 3:

        fixed_letters = _reduce_extra_letters(
            fixed_letters
        )


    plate = (
        prefix
        + fixed_letters
        + fixed_number
    )


    # --------------------------------------------------------
    # ตรวจ Regex
    # --------------------------------------------------------

    if PLATE_PATTERN.fullmatch(
        plate
    ):

        return plate


    return None


# ============================================================
# SPLIT NUMBER ZONE
# ============================================================

def _split_number_letter_zone(raw):

    idx = len(raw)


    for i in range(
        len(raw) - 1,
        -1,
        -1
    ):

        ch = raw[i]


        if (
            ch.isdigit()
            or ch in DIGIT_MAP
        ):

            idx = i

        else:

            break


    if idx == len(raw):

        return (
            None,
            raw
        )


    return (
        raw[idx:],
        raw[:idx]
    )


# ============================================================
# REPAIR DIGITS
# ============================================================

def _repair_digits(text):

    return "".join(
        DIGIT_MAP.get(
            c,
            c
        )
        for c in text
    )


# ============================================================
# REPAIR THAI LETTERS
# ============================================================

def _repair_letters(text):
    """คืน (พยัญชนะไทยที่ซ่อมแล้ว, จำนวนตัวที่ทิ้งเพราะแปลไม่ออก)

    อัพเดทจาก ocr_new.py: ใช้ PLATE_ONLY_MAP (ไทย -> ไทย เช่น า -> ว) ร่วมกับ
    THAI_MAP เฉพาะตรงนี้ (โซนตัวอักษรของเลขทะเบียน) เท่านั้น — clean_province_text
    ใช้แค่ THAI_MAP เพื่อไม่ให้ า -> ว รั่วไปโดนชื่อจังหวัด
    """

    fixed = "".join(
        PLATE_ONLY_MAP.get(
            c,
            THAI_MAP.get(c, c)
        )
        for c in text
    )

    kept = "".join(
        c
        for c in fixed
        if c in VALID_THAI_CONSONANTS
    )

    return kept, len(fixed) - len(kept)


# ============================================================
# REDUCE EXTRA LETTERS
# ============================================================

def _reduce_extra_letters(letters):

    if not letters:
        return ""


    deduped = letters[0]


    for c in letters[1:]:

        if c != deduped[-1]:

            deduped += c


    if len(deduped) <= 3:

        return deduped


    for drop_count in range(
        1,
        len(deduped) - 2
    ):

        for positions in combinations(
            range(
                len(deduped)
            ),
            drop_count
        ):

            candidate = "".join(
                c
                for i, c in enumerate(
                    deduped
                )
                if i not in positions
            )


            if (
                1 <= len(candidate) <= 3
                and all(
                    c in VALID_THAI_CONSONANTS
                    for c in candidate
                )
            ):

                return candidate


    return deduped[:3]


# ============================================================
# CLEAN PROVINCE
# ============================================================

def clean_province_text(text):

    text = text.strip()


    for old, new in THAI_MAP.items():

        text = text.replace(
            old,
            new
        )


    text = re.sub(
        r'[^ก-๙]',
        '',
        text
    )


    return text


# ============================================================
# FUZZY MATCHING PROVINCE
# ============================================================

def match_province(
    text,
    threshold
):

    if not text:
        return (
            None,
            0
        )


    result = process.extractOne(
        text,
        THAI_PROVINCES,
        scorer=fuzz.token_set_ratio
    )


    if result is None:

        return (
            None,
            0
        )


    match, score, _ = result


    if score >= threshold:

        return (
            match,
            score
        )


    return (
        None,
        score
    )


# ============================================================
# PLATE VOTING (อัพเดทจาก ocr_new.py)
# ============================================================
# โหวตทั้งสตริงแทนการโหวตทีละตัวอักษรแบบเดิม — items = [(ข้อความที่ OCR ถอดได้,
# คะแนน conf)] คะแนนต่อสตริง = ผลรวมน้ำหนักของทุกเฟรมที่อ่านได้เป็นสตริงนั้น
# สตริงที่ผิดฟอร์แมตป้ายไทยหักเหลือ BAD_PATTERN_WEIGHT — เฟรมเดียวที่อ่านมั่ว
# ด้วย conf สูงจึงแพ้ป้ายฟอร์แมตถูกที่โผล่ซ้ำหลายเฟรม
def plate_vote(items):

    tally = Counter()

    for text, weight in items:
        if text:
            bonus = 1.0 if PLATE_PATTERN.fullmatch(text) else BAD_PATTERN_WEIGHT
            tally[text] += weight * bonus

    if not tally:
        return "", 0.0

    best, score = tally.most_common(1)[0]

    # ทุกเฟรมผิดฟอร์แมตหมด = น่าจะพลาดคนละตัวอักษร ลองประกอบใหม่ทีละตัวจากเสียงส่วนใหญ่
    if not PLATE_PATTERN.fullmatch(best):
        merged = character_vote([t for t, _ in items])
        if PLATE_PATTERN.fullmatch(merged):
            return merged, score

    return best, score


# ============================================================
# CHARACTER VOTING
# ============================================================

def character_vote(texts):

    texts = [
        t
        for t in texts
        if t
    ]


    if not texts:
        return ""


    max_len = max(
        len(t)
        for t in texts
    )


    result = ""


    for i in range(
        max_len
    ):

        chars = [
            t[i]
            for t in texts
            if i < len(t)
        ]


        if chars:

            result += Counter(
                chars
            ).most_common(
                1
            )[0][0]


    return result


# ============================================================
# PROVINCE VOTING
# ============================================================

def province_vote(provinces):

    provinces = [
        p
        for p in provinces
        if p
    ]


    if not provinces:
        return None


    return Counter(
        provinces
    ).most_common(
        1
    )[0][0]


# ============================================================
# READ ONE PLATE
# ============================================================

def read_plate(
    image_source,
    province_threshold=60,
    conf_threshold=0.6,
    yolo2_conf=YOLO2_CONF
):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    if isinstance(
        image_source,
        (str, Path)
    ):

        img = cv2.imread(
            str(image_source)
        )


        if img is None:

            raise FileNotFoundError(
                f"ไม่พบไฟล์: {image_source}"
            )


    elif isinstance(
        image_source,
        np.ndarray
    ):

        img = image_source


    else:

        raise TypeError(
            "image_source ต้องเป็น str, Path หรือ np.ndarray"
        )


    # ========================================================
    # ADAPTIVE GAMMA + CLAHE (ปิดอยู่โดยดีฟอลต์ — เปิดด้วย DETECT_ENHANCE=1)
    # ========================================================

    if DETECT_ENHANCE:

        img = enhance_image(
            img
        )


    # ========================================================
    # YOLO #2
    # ========================================================

    plate_number_images, province_images = (
        detect_plate_parts(
            img,
            conf=yolo2_conf
        )
    )


    # ========================================================
    # OCR NUMBER
    # ========================================================

    license_results = []


    for detection in plate_number_images:

        crop = detection["image"]


        results = ocr_image(
            crop,
            conf_threshold
        )


        for text, score in results:

            license_id = clean_plate_number(
                text
            )


            if license_id:

                license_results.append(
                    (license_id, score)
                )


    # ========================================================
    # OCR PROVINCE
    # ========================================================

    province_results = []

    # เก็บ ocr score (rec_score) แยกตามจังหวัดที่ match ได้ ไว้คำนวณ
    # "ความมั่นใจเฉลี่ย" ของจังหวัดที่ชนะโหวตในขั้นตอนถัดไป — แยกจาก
    # province_results (list ชื่อจังหวัดล้วน) เพราะ province_vote() เดิม
    # ยังต้องรับ list ของสตริงเหมือนก่อน ไม่แตะ signature ของมัน
    province_score_map = {}


    for detection in province_images:

        crop = detection["image"]


        results = ocr_image(
            crop,
            conf_threshold
        )


        for text, score in results:

            province_raw = clean_province_text(
                text
            )


            province, prov_score = match_province(
                province_raw,
                threshold=province_threshold
            )


            if province:

                province_results.append(
                    province
                )

                province_score_map.setdefault(
                    province, []
                ).append(score)


    # ========================================================
    # FINAL VOTING
    # ========================================================

    # อัพเดทจาก ocr_new.py: ใช้ plate_vote (โหวตทั้งสตริง ถ่วงน้ำหนักด้วย conf +
    # โบนัสฟอร์แมตถูก) แทน character_vote เดิม (โหวตทีละตัวอักษรแบบไม่สนใจฟอร์แมต/conf)
    final_license, _license_score = plate_vote(
        license_results
    )

    final_province = province_vote(
        province_results
    )


    # ========================================================
    # CONFIDENCE (ความมั่นใจเฉลี่ย)
    # ========================================================
    # _license_score จาก plate_vote คือ "คะแนนโหวตสะสม" (ถ่วงน้ำหนัก x โบนัส
    # ฟอร์แมต) ไม่ใช่ความมั่นใจดิบ ใช้รายงานตรงๆ ไม่ได้ (ค่าอาจเกิน 1.0)
    # เอา rec_score ดิบเฉลี่ยเฉพาะรอบที่ทายตรงกับผลโหวตสุดท้ายแทน

    license_scores = [
        score
        for text, score in license_results
        if text == final_license
    ]

    license_confidence = (
        sum(license_scores) / len(license_scores)
        if license_scores
        else 0.0
    )


    province_scores = province_score_map.get(
        final_province,
        []
    )

    province_confidence = (
        sum(province_scores) / len(province_scores)
        if province_scores
        else 0.0
    )


    return (
        final_license,
        final_province,
        license_confidence,
        province_confidence
    )


# ============================================================
# READ MULTIPLE FRAMES
# ============================================================

def read_plate_sequence(
    images,
    province_threshold=60,
    conf_threshold=0.6,
    yolo2_conf=YOLO2_CONF
):

    license_results = []
    province_results = []


    for img in images:

        license_id, province, _lic_conf, _prov_conf = read_plate(
            img,
            province_threshold,
            conf_threshold,
            yolo2_conf
        )


        if license_id:

            license_results.append(
                (license_id, 1.0)
            )


        if province:

            province_results.append(
                province
            )


    final_license, _license_score = plate_vote(
        license_results
    )

    final_province = province_vote(
        province_results
    )


    return (
        final_license,
        final_province
    )


# ============================================================
# READ FOLDER
# ============================================================

def read_folder(
    folder_path,
    province_threshold=60,
    conf_threshold=0.6,
    debug=True,
    yolo2_conf=YOLO2_CONF
):

    folder = Path(
        folder_path
    )


    if not folder.exists():

        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์: {folder_path}"
        )


    image_files = sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXT
    )


    if not image_files:

        print(
            f"⚠️ ไม่พบไฟล์ภาพใน {folder_path}"
        )

        return []


    print(
        f"🔍 พบ {len(image_files)} ภาพใน {folder_path}\n"
    )


    all_results = []


    for i, img_path in enumerate(
        image_files,
        1
    ):

        print(
            f"[{i}/{len(image_files)}] {img_path.name}"
        )


        try:

            plate_number, province, plate_confidence, province_confidence = read_plate(
                str(img_path),
                province_threshold,
                conf_threshold,
                yolo2_conf
            )


            valid_plate = bool(
                plate_number
                and PLATE_PATTERN.fullmatch(
                    plate_number
                )
            )


            result = {
                "file": img_path.name,
                "plate_number": plate_number,
                "province": province,
                "valid_plate": valid_plate,
                "plate_confidence": plate_confidence,
                "province_confidence": province_confidence
            }


            if debug:

                print(
                    f"   → เลขทะเบียน : "
                    f"{plate_number or '(ไม่พบ)'} "
                    f"(conf {plate_confidence:.2f})"
                )

                print(
                    f"   → จังหวัด    : "
                    f"{province or '(ไม่พบ)'} "
                    f"(conf {province_confidence:.2f})"
                )

                print(
                    f"   → รูปแบบป้าย : "
                    f"{'✓ ถูกต้อง' if valid_plate else '✗ ไม่ผ่าน Regex'}"
                )


            all_results.append(
                result
            )


        except Exception as e:

            print(
                f"   ❌ Error: {e}"
            )


            all_results.append({
                "file": img_path.name,
                "plate_number": "",
                "province": None,
                "valid_plate": False,
                "plate_confidence": 0.0,
                "province_confidence": 0.0,
                "error": str(e)
            })


    return all_results


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(csv_path):

    gt_path = Path(
        csv_path
    )


    if not gt_path.exists():

        raise FileNotFoundError(
            f"ไม่พบไฟล์เฉลย: {csv_path}"
        )


    gt = {}


    with open(
        gt_path,
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(
            f
        )


        for row in reader:

            fname = row["file"].strip()


            gt[fname] = {
                "plate_number": row["plate_number"].strip(),
                "province": row["province"].strip()
            }


    return gt


# ============================================================
# EVALUATE FOLDER
# ============================================================

def evaluate_folder(
    folder_path,
    ground_truth_csv,
    **kwargs
):

    gt = load_ground_truth(
        ground_truth_csv
    )


    results = read_folder(
        folder_path,
        **kwargs
    )


    stats = {
        "total": 0,
        "correct_all": 0,
        "correct_plate_only": 0,
        "correct_province_only": 0,
        "correct_none": 0,
        "no_ground_truth": 0
    }


    details = []


    for r in results:

        fname = r["file"]


        if fname not in gt:

            stats["no_ground_truth"] += 1


            details.append({
                "file": fname,
                "expected_plate": "",
                "predicted_plate": r.get(
                    "plate_number",
                    ""
                ),
                "expected_province": "",
                "predicted_province": r.get(
                    "province"
                ),
                "status": "no_ground_truth"
            })


            continue


        stats["total"] += 1


        expected = gt[fname]


        predicted_plate = (
            r.get(
                "plate_number",
                ""
            )
            or ""
        )


        predicted_province = r.get(
            "province"
        )


        plate_ok = (
            predicted_plate
            == expected["plate_number"]
        )


        province_ok = (
            predicted_province
            == expected["province"]
        )


        if plate_ok and province_ok:

            status = "correct_all"


        elif plate_ok:

            status = "correct_plate_only"


        elif province_ok:

            status = "correct_province_only"


        else:

            status = "correct_none"


        stats[status] += 1


        details.append({
            "file": fname,
            "expected_plate": expected[
                "plate_number"
            ],
            "predicted_plate": predicted_plate,
            "expected_province": expected[
                "province"
            ],
            "predicted_province": predicted_province,
            "status": status
        })


    return {
        "stats": stats,
        "details": details
    }


# ============================================================
# EXPORT RESULT CSV
# ============================================================

def export_eval_csv(
    eval_result,
    out_path="eval_report.csv"
):

    details = eval_result[
        "details"
    ]


    if not details:

        print(
            "⚠️ ไม่มีข้อมูลให้เซฟ"
        )

        return


    fieldnames = [
        "file",
        "expected_plate",
        "predicted_plate",
        "expected_province",
        "predicted_province",
        "status"
    ]


    with open(
        out_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )


        writer.writeheader()


        for d in details:

            writer.writerow(
                d
            )


    print(
        f"\n📄 บันทึกรายงานที่:"
    )

    print(
        Path(out_path).resolve()
    )


# ============================================================
# SAVE READ RESULT ONLY
# ============================================================

def export_read_results(
    results,
    out_path="plate_results.csv"
):
    """
    บันทึกผลที่ระบบอ่านได้

    CSV:
    file,plate_number,province,valid_plate,plate_confidence,province_confidence
    """

    if not results:

        print(
            "⚠️ ไม่มีข้อมูลให้บันทึก"
        )

        return


    fieldnames = [
        "file",
        "plate_number",
        "province",
        "valid_plate",
        "plate_confidence",
        "province_confidence"
    ]


    with open(
        out_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )


        writer.writeheader()


        for result in results:

            writer.writerow({
                "file": result.get(
                    "file",
                    ""
                ),
                "plate_number": result.get(
                    "plate_number",
                    ""
                ),
                "province": result.get(
                    "province"
                ) or "",
                "valid_plate": result.get(
                    "valid_plate",
                    False
                ),
                "plate_confidence": result.get(
                    "plate_confidence",
                    0.0
                ),
                "province_confidence": result.get(
                    "province_confidence",
                    0.0
                )
            })


    print(
        f"\n📄 บันทึกผลการอ่านป้ายที่:"
    )

    print(
        Path(out_path).resolve()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    folder = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"D:\license_plate_and_car_detection\validate_img"
    )


    gt_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else None
    )


    # ========================================================
    # MODE 1:
    # มี CSV เฉลย
    # ========================================================

    if gt_csv:

        eval_result = evaluate_folder(
            folder,
            gt_csv,
            province_threshold=60,
            conf_threshold=0.6,
            debug=True
        )


        export_eval_csv(
            eval_result,
            "eval_report.csv"
        )


    # ========================================================
    # MODE 2:
    # อ่านป้ายอย่างเดียว
    # ========================================================

    else:

        results = read_folder(
            folder,
            province_threshold=60,
            conf_threshold=0.6,
            debug=True
        )


        export_read_results(
            results,
            "plate_results.csv"
        )