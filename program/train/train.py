from ultralytics import YOLO
import torch

def main():

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"D:\license_plate_and_car_detection\dataset\dataset_6\data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
    )

if __name__ == "__main__":
    main()