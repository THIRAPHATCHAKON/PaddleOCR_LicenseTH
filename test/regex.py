import re


# ============================================================
# PLATE PATTERN
# ============================================================

PLATE_PATTERN = re.compile(
    r'^[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}$'
)


# ============================================================
# VALIDATE PLATE
# ============================================================

def validate_plate(plate):
    """
    ตรวจสอบว่าเลขทะเบียนตรงตามรูปแบบที่กำหนดหรือไม่
    """

    plate = plate.strip().replace(" ", "")

    return bool(PLATE_PATTERN.fullmatch(plate))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("LICENSE PLATE PATTERN VALIDATION")
    print("=" * 55)

    print("""
รูปแบบที่ระบบยอมรับ:

[ตัวเลขนำหน้า 0-1 ตัว]
        +
[ตัวอักษรไทย 1-3 ตัว]
        +
[ตัวเลข 1-4 ตัว]

ตัวอย่าง:
ก1234
กข1234
1กข1234
2กก123
""")

    while True:

        plate = input(
            "กรอกเลขทะเบียน (exit เพื่อออก): "
        ).strip()

        if plate.lower() == "exit":
            break

        # ลบช่องว่าง
        cleaned = plate.replace(" ", "")

        # ตรวจสอบ
        valid = validate_plate(cleaned)

        print("-" * 55)
        print(f"OCR Input : {plate}")
        print(f"Cleaned   : {cleaned}")

        if valid:
            print("Pattern   : MATCH")
            print("Result    : PASS ✓")
        else:
            print("Pattern   : NOT MATCH")
            print("Result    : REJECT ✗")

        print("-" * 55)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()