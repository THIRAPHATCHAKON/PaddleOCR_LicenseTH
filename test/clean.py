# ============================================================
# CONFUSION MAP SIMULATION
# ============================================================


# ============================================================
# CONFUSION MAP
# ============================================================

# อังกฤษ / ตัวเลข -> ตัวอักษรไทย
# ใช้ได้กับตัวอักษรทะเบียน และการ clean จังหวัด
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


# ไทย -> ไทย
# ใช้เฉพาะตัวอักษรของเลขทะเบียน
PLATE_ONLY_MAP = {
    "า": "ว",
    "ฤ": "ฎ",
}


# อังกฤษ -> ตัวเลข
# ใช้เฉพาะโซนตัวเลขทะเบียน
DIGIT_MAP = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",

    "I": "1",
    "l": "1",
    "i": "1",
    "m": "1",

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
}


# ============================================================
# APPLY MAP
# ============================================================

def apply_map(text, mapping):
    """
    แทนที่ตัวอักษรทีละตัวตาม Dictionary
    """
    return "".join(
        mapping.get(char, char)
        for char in text
    )


# ============================================================
# PLATE LETTER CORRECTION
# ============================================================

def clean_plate_letters(text):
    """
    สำหรับโซนตัวอักษรทะเบียน

    ใช้:
        THAI_MAP
        +
        PLATE_ONLY_MAP
    """

    # Step 1
    text = apply_map(text, THAI_MAP)

    # Step 2
    text = apply_map(text, PLATE_ONLY_MAP)

    return text


# ============================================================
# PLATE DIGIT CORRECTION
# ============================================================

def clean_plate_digits(text):
    """
    สำหรับโซนตัวเลขทะเบียน

    ใช้เฉพาะ DIGIT_MAP
    """

    return apply_map(text, DIGIT_MAP)


# ============================================================
# PROVINCE CORRECTION
# ============================================================

def clean_province_text(text):
    """
    สำหรับชื่อจังหวัด

    ใช้เฉพาะ THAI_MAP

    สำคัญ:
    ห้ามใช้ PLATE_ONLY_MAP
    เพราะอาจทำให้ชื่อจังหวัดเสีย
    เช่น

        ตาก
         ↓

    ถ้าใช้ {"า": "ว"}

        ตวก  ❌
    """

    return apply_map(text, THAI_MAP)


# ============================================================
# SHOW CHARACTER CHANGES
# ============================================================

def show_changes(original, corrected):

    print("\nการแทนที่:")

    changed = False

    for old, new in zip(original, corrected):

        if old != new:
            print(f"    {old}  ->  {new}")
            changed = True

    if not changed:
        print("    ไม่มีการแทนที่")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("OCR CONFUSION MAP SIMULATION")
    print("=" * 60)

    print("""
เลือกโหมด

1 = ตัวอักษรทะเบียน
2 = ตัวเลขทะเบียน
3 = จังหวัด
4 = ออกจากโปรแกรม
""")

    while True:

        mode = input("เลือกโหมด: ").strip()

        # ----------------------------------------------------
        # PLATE LETTER
        # ----------------------------------------------------

        if mode == "1":

            text = input(
                "\nOCR ตัวอักษรทะเบียน: "
            ).strip()

            corrected = clean_plate_letters(text)

            print("\n" + "-" * 50)
            print("ประเภท      : ตัวอักษรทะเบียน")
            print(f"OCR Input    : {text}")
            print(f"Corrected    : {corrected}")

            show_changes(
                text,
                corrected
            )

            print("-" * 50)

        # ----------------------------------------------------
        # PLATE DIGIT
        # ----------------------------------------------------

        elif mode == "2":

            text = input(
                "\nOCR ตัวเลขทะเบียน: "
            ).strip()

            corrected = clean_plate_digits(text)

            print("\n" + "-" * 50)
            print("ประเภท      : ตัวเลขทะเบียน")
            print(f"OCR Input    : {text}")
            print(f"Corrected    : {corrected}")

            show_changes(
                text,
                corrected
            )

            print("-" * 50)

        # ----------------------------------------------------
        # PROVINCE
        # ----------------------------------------------------

        elif mode == "3":

            text = input(
                "\nOCR จังหวัด: "
            ).strip()

            corrected = clean_province_text(text)

            print("\n" + "-" * 50)
            print("ประเภท      : จังหวัด")
            print(f"OCR Input    : {text}")
            print(f"Corrected    : {corrected}")

            show_changes(
                text,
                corrected
            )

            print("-" * 50)

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif mode == "4":

            print("\nจบการทำงาน")
            break

        else:

            print(
                "กรุณาเลือก 1, 2, 3 หรือ 4"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()