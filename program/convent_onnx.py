from ultralytics import YOLO

model = YOLO(
    r"D:\license_plate_and_car_detection\runs\detect\YOLO\License_Province3_320\weights\best.pt"
)

model.export(
    format="onnx",
    imgsz=320,
    opset=12,
    simplify=True,
    dynamic=False
)