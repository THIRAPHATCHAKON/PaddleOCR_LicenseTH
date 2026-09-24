import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = "plate_1.jpg"
OUTPUT_IMAGE = "plate_perspective.jpg"

# ต้องมีพื้นที่อย่างน้อย 50% ของ crop เดิม
MIN_AREA_RATIO = 0.20


# ============================================================
# ORDER POINTS
# ============================================================

def order_points(pts):
    """
    เรียง 4 มุมเป็น:

    0 = Top-Left
    1 = Top-Right
    2 = Bottom-Right
    3 = Bottom-Left
    """

    rect = np.zeros((4, 2), dtype=np.float32)

    # x + y
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # y - x
    diff = np.diff(pts, axis=1).reshape(-1)

    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


# ============================================================
# PERSPECTIVE PLATE
# ============================================================

def perspective_plate(plate):

    if plate is None or plate.size == 0:
        return plate

    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # GAUSSIAN BLUR
    # ========================================================

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ========================================================
    # OTSU THRESHOLD
    # ========================================================

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        print("❌ No contour")

        return plate

    # ========================================================
    # BIGGEST CONTOUR
    # ========================================================

    contour = max(
        contours,
        key=cv2.contourArea
    )

    # ========================================================
    # MIN AREA RECTANGLE
    # ========================================================

    rect = cv2.minAreaRect(
        contour
    )

    center, size, angle = rect

    print("\n========== RECT INFO ==========")

    print(
        f"Center : "
        f"({center[0]:.2f}, {center[1]:.2f})"
    )

    print(
        f"Size   : "
        f"{size[0]:.2f} x {size[1]:.2f}"
    )

    print(
        f"Angle  : {angle:.2f}°"
    )

    # ========================================================
    # GET 4 CORNERS
    # ========================================================

    box = cv2.boxPoints(
        rect
    )

    box = np.float32(
        box
    )

    box = order_points(
        box
    )

    print("\n4 Corners:")

    print(box)

    # ========================================================
    # CALCULATE WIDTH
    # ========================================================

    widthA = np.linalg.norm(
        box[2] - box[3]
    )

    widthB = np.linalg.norm(
        box[1] - box[0]
    )

    maxWidth = int(
        max(
            widthA,
            widthB
        )
    )

    # ========================================================
    # CALCULATE HEIGHT
    # ========================================================

    heightA = np.linalg.norm(
        box[1] - box[2]
    )

    heightB = np.linalg.norm(
        box[0] - box[3]
    )

    maxHeight = int(
        max(
            heightA,
            heightB
        )
    )

    # ========================================================
    # AREA CHECK
    # ========================================================

    h, w = plate.shape[:2]

    original_area = h * w
    detected_area = maxWidth * maxHeight

    area_ratio = (
        detected_area / original_area
        if original_area > 0
        else 0
    )

    print(
        f"\nOriginal size : {w} x {h}"
    )

    print(
        f"Detected size : {maxWidth} x {maxHeight}"
    )

    print(
        f"Area ratio    : {area_ratio * 100:.2f}%"
    )

    if (
        maxWidth < 2
        or maxHeight < 2
        or area_ratio < MIN_AREA_RATIO
    ):

        print(
            "\n⚠️ Rectangle ไม่น่าเชื่อถือ"
        )

        print(
            "→ คืนภาพต้นฉบับ"
        )

        return plate

    # ========================================================
    # DESTINATION POINTS
    # ========================================================

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ],
        dtype=np.float32
    )

    # ========================================================
    # PERSPECTIVE MATRIX
    # ========================================================

    M = cv2.getPerspectiveTransform(
        box,
        dst
    )

    # ========================================================
    # WARP
    # ========================================================

    warp = cv2.warpPerspective(
        plate,
        M,
        (
            maxWidth,
            maxHeight
        )
    )

    print(
        "\n✓ Perspective Transform สำเร็จ"
    )

    return warp


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # อ่านภาพ
    image = cv2.imread(
        INPUT_IMAGE
    )

    if image is None:

        raise FileNotFoundError(
            f"ไม่พบรูป: {INPUT_IMAGE}"
        )

    print(
        f"Input : {Path(INPUT_IMAGE).resolve()}"
    )

    # Perspective Transform
    result = perspective_plate(
        image
    )

    # Save
    success = cv2.imwrite(
        OUTPUT_IMAGE,
        result
    )

    if success:

        print(
            f"\nOutput: "
            f"{Path(OUTPUT_IMAGE).resolve()}"
        )

    else:

        print(
            "\n❌ Save failed"
        )