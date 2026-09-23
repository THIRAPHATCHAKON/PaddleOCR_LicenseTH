import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"" # Path

INPUT_DIR = Path("input_images")
ENHANCED_DIR = Path("enhanced_images")
OUTPUT_DIR = Path("output_images")
COMPARISON_DIR = Path("comparison_images")

CONFIDENCE = 0.40
TARGET_CLASS = 1

TARGET_BRIGHTNESS = 130
MIN_GAMMA = 0.4
MAX_GAMMA = 2.5

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

PLATE_WIDTH = 320
PLATE_HEIGHT = 120

ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)


def adaptive_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    mean_normalized = max(mean_brightness / 255.0, 0.01)
    target_normalized = TARGET_BRIGHTNESS / 255.0

    gamma = np.log(target_normalized) / np.log(mean_normalized)
    gamma = np.clip(gamma, MIN_GAMMA, MAX_GAMMA)

    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)

    return cv2.LUT(image, table)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_image(image):
    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy.ndarray")

    gamma_image = adaptive_gamma(image)
    return apply_clahe(gamma_image)


def order_points(points):
    points = np.array(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)

    result[0] = points[np.argmin(s)]
    result[1] = points[np.argmin(diff)]
    result[2] = points[np.argmax(s)]
    result[3] = points[np.argmax(diff)]

    return result


def perspective_rectify(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_area = plate.shape[0] * plate.shape[1]
    best_quad = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < plate_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 4:
            quad_area = cv2.contourArea(approx)

            if quad_area > best_area:
                best_area = quad_area
                best_quad = approx.reshape(4, 2)

    # Perspective transform
    if best_quad is not None:
        points = order_points(best_quad)

        destination = np.array([
            [0, 0],
            [PLATE_WIDTH - 1, 0],
            [PLATE_WIDTH - 1, PLATE_HEIGHT - 1],
            [0, PLATE_HEIGHT - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(points, destination)
        corrected = cv2.warpPerspective(plate, matrix, (PLATE_WIDTH, PLATE_HEIGHT))

        return corrected, "perspective"

    # Rotation fallback
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > plate_area * 0.10:
            rect = cv2.minAreaRect(largest)
            angle = rect[2]
            width, height = rect[1]

            if width < height:
                angle += 90

            center = (plate.shape[1] // 2, plate.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated = cv2.warpAffine(
                plate,
                rotation_matrix,
                (plate.shape[1], plate.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            rotated = cv2.resize(rotated, (PLATE_WIDTH, PLATE_HEIGHT), interpolation=cv2.INTER_CUBIC)

            return rotated, "rotation"

    # Resize fallback
    resized = cv2.resize(plate, (PLATE_WIDTH, PLATE_HEIGHT), interpolation=cv2.INTER_CUBIC)

    return resized, "resize"


image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in image_extensions)

for image_path in image_files:
    original = cv2.imread(str(image_path))

    if original is None:
        continue

    # Image enhancement
    enhanced = enhance_image(original)
    cv2.imwrite(str(ENHANCED_DIR / image_path.name), enhanced)

    # License plate detection
    results = model.predict(
        source=enhanced,
        conf=CONFIDENCE,
        classes=[TARGET_CLASS],
        verbose=False
    )

    plate_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(enhanced.shape[1], x2)
        y2 = min(enhanced.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        plate = enhanced[y1:y2, x1:x2].copy()

        if plate.size == 0:
            continue

        # Perspective correction
        corrected, method = perspective_rectify(plate)

        plate_count += 1

        output_name = f"{image_path.stem}_plate_{plate_count}_{method}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / output_name), corrected)

        # Comparison image
        original_plate = cv2.resize(plate, (PLATE_WIDTH, PLATE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        comparison = np.hstack([original_plate, corrected])

        cv2.putText(
            comparison,
            "Detected Plate",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            comparison,
            f"Corrected: {method}",
            (PLATE_WIDTH + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        comparison_name = f"{image_path.stem}_plate_{plate_count}_comparison.jpg"
        cv2.imwrite(str(COMPARISON_DIR / comparison_name), comparison)

print(f"Completed: {len(image_files)} images")
