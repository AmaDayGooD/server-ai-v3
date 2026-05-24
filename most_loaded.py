import os
from ultralytics import YOLO

MODEL_PATH = "runs/obb/parking_obb/weights/best.pt"

def get_top_k_loaded_images(images_dir: str, k=3):
    model = YOLO(MODEL_PATH)

    scored = []

    for filename in os.listdir(images_dir):

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(images_dir, filename)

        results = model(path)

        car_count = 0

        for r in results:
            if r.obb is None:
                continue
            car_count += len(r.obb.xyxyxyxy)

        scored.append((path, car_count))
        print(f"{filename}: {car_count}")

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:k]