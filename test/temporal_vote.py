import re
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

# รูปแบบป้ายเดียวกับโค้ดปัจจุบัน
PLATE_PATTERN = re.compile(
    r'^[0-9]{0,1}[\u0E01-\u0E4E]{1,10}[0-9]{1,4}$'
)

# ถ้าป้ายผิด Pattern จะเหลือน้ำหนัก 40%
BAD_PATTERN_WEIGHT = 0.4


# ============================================================
# CHARACTER VOTING
# ============================================================

def character_vote(texts):
    """
    Fallback:
    ถ้าผลที่ชนะ Temporal Vote ยังผิด Pattern
    ให้โหวตตัวอักษรทีละตำแหน่ง
    """

    texts = [
        text
        for text in texts
        if text
    ]

    if not texts:
        return ""

    max_len = max(
        len(text)
        for text in texts
    )

    result = ""

    for i in range(max_len):

        chars = [
            text[i]
            for text in texts
            if i < len(text)
        ]

        if chars:
            winner = Counter(chars).most_common(1)[0][0]
            result += winner

    return result


# ============================================================
# TEMPORAL PLATE VOTING
# ============================================================

def plate_vote(items):
    """
    items:

    [
        ("กค4678", 1.0),
        ("กค4678", 1.0),
        ("กศ4678", 1.0)
    ]

    ใน read_plate_sequence() ของโค้ดจริง
    แต่ละเฟรมถูกกำหนด weight = 1.0
    """

    tally = Counter()

    print("\n" + "=" * 65)
    print("TEMPORAL VOTING")
    print("=" * 65)

    # --------------------------------------------------------
    # นับคะแนนแต่ละเฟรม
    # --------------------------------------------------------

    for frame_number, (text, weight) in enumerate(items, 1):

        if not text:
            continue

        valid = bool(
            PLATE_PATTERN.fullmatch(text)
        )

        # ถ้าถูก Pattern = น้ำหนักเต็ม
        # ถ้าผิด Pattern = เหลือ 40%
        bonus = (
            1.0
            if valid
            else BAD_PATTERN_WEIGHT
        )

        vote_score = weight * bonus

        tally[text] += vote_score

        print(
            f"Frame {frame_number:02d} | "
            f"{text:<12} | "
            f"Pattern: {'PASS' if valid else 'FAIL':<4} | "
            f"Weight: {weight:.2f} | "
            f"Vote: {vote_score:.2f}"
        )

    # --------------------------------------------------------
    # ไม่มีข้อมูล
    # --------------------------------------------------------

    if not tally:
        return "", 0.0

    # --------------------------------------------------------
    # แสดงคะแนนรวม
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("คะแนนรวมจากทุก Frame")
    print("-" * 65)

    for text, score in tally.most_common():

        print(
            f"{text:<15} -> {score:.2f}"
        )

    # --------------------------------------------------------
    # หาผู้ชนะ
    # --------------------------------------------------------

    best, score = tally.most_common(1)[0]

    print("\n" + "-" * 65)
    print(f"ผู้ชนะเบื้องต้น : {best}")
    print(f"คะแนนสะสม      : {score:.2f}")

    # --------------------------------------------------------
    # FALLBACK CHARACTER VOTE
    # --------------------------------------------------------

    if not PLATE_PATTERN.fullmatch(best):

        print("\nผู้ชนะไม่ผ่าน Regex")
        print("→ ทำ Character Voting")

        merged = character_vote(
            [
                text
                for text, _ in items
            ]
        )

        print(
            f"Character Vote  : {merged}"
        )

        if PLATE_PATTERN.fullmatch(merged):

            print(
                "Character Vote ผ่าน Regex ✓"
            )

            return merged, score

        print(
            "Character Vote ไม่ผ่าน Regex ✗"
        )

    return best, score


# ============================================================
# PROVINCE TEMPORAL VOTING
# ============================================================

def province_vote(provinces):

    provinces = [
        province
        for province in provinces
        if province
    ]

    if not provinces:
        return None

    tally = Counter(provinces)

    print("\n" + "=" * 65)
    print("PROVINCE TEMPORAL VOTING")
    print("=" * 65)

    for province, count in tally.most_common():

        print(
            f"{province:<20} -> {count} votes"
        )

    return tally.most_common(1)[0][0]


# ============================================================
# TEMPORAL VOTE SIMULATION
# ============================================================

def temporal_vote(frames):

    """
    จำลองผลจากหลาย Frame

    แต่ละ Frame:
        (plate, province)

    เช่น:
        ("กค4678", "ขอนแก่น")
    """

    license_results = []
    province_results = []

    print("\nผลจากแต่ละ Frame")
    print("=" * 65)

    for i, (plate, province) in enumerate(
        frames,
        1
    ):

        print(
            # f"Frame {i:02d} -> "
            f"{plate:<12} "
            f"{province}"
        )

        # เหมือน read_plate_sequence()
        if plate:
            license_results.append(
                (plate, 1.0)
            )

        if province:
            province_results.append(
                province
            )

    # ========================================================
    # PLATE VOTE
    # ========================================================

    final_plate, vote_score = plate_vote(
        license_results
    )

    # ========================================================
    # PROVINCE VOTE
    # ========================================================

    final_province = province_vote(
        province_results
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL TEMPORAL VOTE")
    print("=" * 65)

    print(
        f"{final_plate}"
    )

    print(
        f"{final_province}"
    )

    print(
        f"Vote Score        : {vote_score:.2f}"
    )

    print("=" * 65)

    return final_plate, final_province


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # สมมติเป็นผล OCR จากรถคันเดียวกันหลาย Frame
    frames = [

        ("ขษ2202", "ขอนแก่น"),   # Frame 1 ถูก
        ("ขษ2202", "ขอนแก่น"),   # Frame 2 ถูก
        ("ขษ2202", "กรุงเทพมหานคร"),   # Frame 3 OCR ตัวอักษรผิด
        ("ขบ2202", "ขอนแก่น"),   # Frame 4 ถูก
        ("ขษ2102", "ขอนแก่น"),   # Frame 5 ผิด Pattern
        ("ขษ2202", "กรุงเทพมหานคร"),   # Frame 6 ถูก
        ("ขษ2202", "ขอนแก่น"),   # Frame 7 ถูก

    ]

    temporal_vote(frames)