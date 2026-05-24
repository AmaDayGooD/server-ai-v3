from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="parking_obb"
)

metrics = model.val()

print(metrics)
