from ultralytics import YOLO
import torch
def main():
    
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        
    model = YOLO(
        r"D:\license_plate_and_car_detection\models\train-13\weights\best.pt"
    )

    results = model.train(
        data=r"D:\license_plate_and_car_detection\dataset_13\data.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        save=True,
        project="predict_detect",
        name="fine_ture_2",
        patience=50
    )

if __name__ == "__main__":
    main()