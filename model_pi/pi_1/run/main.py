import cv2
from detect import PlateDetector
from ultralytics import YOLO


def draw(img, detections, model):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls = d["class_id"]

        name = model.names[cls]

        # สี: car = ฟ้า, plate = เขียว
        color = (255, 0, 0) if cls == 0 else (0, 255, 0)

        label = f"{name} {d['conf']:.2f}"

        if cls == 3:
            label += f" | {d['plate']}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    model_path = r"D:\license_plate_and_car_detection\model_pi\pi_1\model\best.pt"

    detector = PlateDetector(model_path, conf=0.3)
    model = YOLO(model_path)

    img_path = r"D:\license_plate_and_car_detection\img\3.jpg"

    img, detections = detector.detect(img_path)

    print("\n===== RESULTS =====")

    for i, d in enumerate(detections):
        print(f"\nObject {i+1}")
        print("Class:", model.names[d["class_id"]])
        print("Conf :", d["conf"])

        if d["class_id"] == 3:
            print("OCR  :", d["ocr_raw"])
            print("Plate:", d["plate"])
            print("Valid:", d["valid"])

    result = draw(img, detections, model)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()