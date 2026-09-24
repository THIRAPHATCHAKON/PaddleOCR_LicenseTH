import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = r"img.jpg"
OUTPUT_IMAGE = r"output_enhanced.jpg"

# ค่าเป้าหมายความสว่าง
TARGET_BRIGHTNESS = 130

# จำกัดค่า Gamma
GAMMA_RANGE = (0.5, 2.0)

# CLAHE
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


# ============================================================
# ADAPTIVE GAMMA
# ============================================================

def adaptive_gamma(image):
    """
    ปรับ Gamma อัตโนมัติจากความสว่างเฉลี่ยของภาพ

    ภาพมืด   -> gamma < 1 -> สว่างขึ้น
    ภาพสว่าง -> gamma > 1 -> มืดลง
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean = max(
        np.mean(gray) / 255.0,
        0.01
    )

    target = TARGET_BRIGHTNESS / 255.0

    gamma = np.log(target) / np.log(mean)

    gamma = np.clip(
        gamma,
        GAMMA_RANGE[0],
        GAMMA_RANGE[1]
    )

    table = (
        ((np.arange(256) / 255.0) ** gamma)
        * 255
    ).astype(np.uint8)

    output = cv2.LUT(image, table)

    return output, gamma


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):

    # BGR -> LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_GRID
    )

    l = clahe.apply(l)

    # รวม Channel กลับ
    lab = cv2.merge(
        (l, a, b)
    )

    # LAB -> BGR
    output = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return output


# ============================================================
# ENHANCE IMAGE
# ============================================================

def enhance_image(image):

    # Step 1: Adaptive Gamma
    gamma_image, gamma = adaptive_gamma(image)

    # Step 2: CLAHE
    output = apply_clahe(gamma_image)

    return output, gamma


# ============================================================
# MAIN
# ============================================================

def main():

    input_path = Path(INPUT_IMAGE)

    # -----------------------------
    # ตรวจสอบไฟล์
    # -----------------------------

    if not input_path.exists():
        print(f"ไม่พบรูป: {INPUT_IMAGE}")
        return

    # -----------------------------
    # อ่านรูป
    # -----------------------------

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        print("ไม่สามารถอ่านรูปได้")
        return

    # ความสว่างก่อนปรับ
    original_gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    original_brightness = np.mean(
        original_gray
    )

    # -----------------------------
    # Enhance
    # -----------------------------

    enhanced, gamma = enhance_image(
        image
    )

    # ความสว่างหลังปรับ
    enhanced_gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )

    enhanced_brightness = np.mean(
        enhanced_gray
    )

    # -----------------------------
    # Save
    # -----------------------------

    cv2.imwrite(
        OUTPUT_IMAGE,
        enhanced
    )

    # -----------------------------
    # แสดงข้อมูล
    # -----------------------------

    print("=" * 50)
    print("Adaptive Gamma + CLAHE")
    print("=" * 50)

    print(
        f"Brightness ก่อน : "
        f"{original_brightness:.2f}"
    )

    print(
        f"Gamma ที่คำนวณ : "
        f"{gamma:.4f}"
    )

    print(
        f"Brightness หลัง : "
        f"{enhanced_brightness:.2f}"
    )

    print(
        f"Target           : "
        f"{TARGET_BRIGHTNESS}"
    )

    print(
        f"Output           : "
        f"{OUTPUT_IMAGE}"
    )

    print("=" * 50)

    # -----------------------------
    # แสดง Before / After
    # -----------------------------

    # ทำให้สองภาพสูงเท่ากัน
    comparison = np.hstack(
        (image, enhanced)
    )

    cv2.imshow(
        "Original | Adaptive Gamma + CLAHE",
        comparison
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()