import cv2
from model_pi.Pi_2.run.detect import PlateDetector, PLATE_CLASS_ID
from ultralytics import YOLO


def draw(img, detections, model):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls = d["class_id"]

        name = model.names[cls]

        # สี: car = ฟ้า, plate = เขียว
        color = (255, 0, 0) if cls == 0 else (0, 255, 0)

        label = f"{name} {d['conf']:.2f}"

        if cls == PLATE_CLASS_ID and d.get("plate_number"):
            label += f" | {d['plate_number']}"
            if d.get("province"):
                label += f" ({d['province']})"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    model_path = r"D:\license_plate_and_car_detection\models\train-12\weights\best.pt"

    detector = PlateDetector(model_path, conf=0.3)
    model = YOLO(model_path)

    img_path = r"D:\license_plate_and_car_detection\img\7.jpg"

    img, detections = detector.detect(img_path)

    print("\n===== RESULTS =====")

    for i, d in enumerate(detections):
        print(f"\nObject {i+1}")
        print("Class:", model.names[d["class_id"]])
        print("Conf :", d["conf"])

        if d["class_id"] == PLATE_CLASS_ID:
            print("Plate   :", d.get("plate_number"))
            print("Province:", d.get("province"), f"(score={d.get('province_score')})")
            print("Valid   :", d.get("valid_plate"))

    result = draw(img, detections, model)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
