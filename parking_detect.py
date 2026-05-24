# Строит карту парковки по самомой "загруженной" фото парковки, сохраняет parking_map.jpg и parking_map.json 

import cv2
import numpy as np
import json
from ultralytics import YOLO

# ==========================================
# ЗАГРУЗКА МОДЕЛИ
# ==========================================
model = YOLO("runs/obb/parking_obb/weights/best.pt")

# ==========================================
# ИЗОБРАЖЕНИЕ
# ==========================================
IMAGE_PATH = "test.jpg"

results = model(IMAGE_PATH)

# ==========================================
# PARKING MAP
# ==========================================
parking_slots = []

for result in results:

    image = result.orig_img.copy()

    if result.obb is None:
        continue

    # ==========================================
    # OBB BOXES
    # ==========================================
    boxes = result.obb.xyxyxyxy.cpu().numpy()

    for idx, box in enumerate(boxes):

        pts = np.array(box, dtype=np.float32)

        # ==========================================
        # РИСУЕМ АВТОМОБИЛЬ
        # ==========================================
        cv2.polylines(
            image,
            [pts.astype(np.int32)],
            isClosed=True,
            color=(0,255,0),
            thickness=2
        )

        # ==========================================
        # СТОРОНЫ
        # ==========================================
        edge1 = pts[1] - pts[0]
        edge2 = pts[2] - pts[1]

        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)

        # ==========================================
        # LONG / SHORT VECTOR
        # ==========================================
        if len1 > len2:
            long_vec = edge1 / len1
            short_vec = edge2 / len2
            car_length = len1
            car_width = len2
        else:
            long_vec = edge2 / len2
            short_vec = edge1 / len1
            car_length = len2
            car_width = len1

        # ==========================================
        # CENTER
        # ==========================================
        center = np.mean(pts, axis=0)

        # ==========================================
        # PARKING SLOT SIZE
        # ==========================================
        slot_length = car_length * 1.15
        slot_width = car_width * 1.2

        # ==========================================
        # BUILD SLOT
        # ==========================================
        p1 = center - long_vec * slot_length/2 - short_vec * slot_width/2
        p2 = center + long_vec * slot_length/2 - short_vec * slot_width/2
        p3 = center + long_vec * slot_length/2 + short_vec * slot_width/2
        p4 = center - long_vec * slot_length/2 + short_vec * slot_width/2

        slot = np.array([p1, p2, p3, p4], dtype=np.int32)

        # ==========================================
        # DRAW SLOT
        # ==========================================
        cv2.polylines(
            image,
            [slot],
            isClosed=True,
            color=(255,0,0),
            thickness=2
        )

        # ==========================================
        # SLOT ID
        # ==========================================
        slot_id = idx + 1

        cv2.putText(
            image,
            f"Slot {slot_id}",
            tuple(slot[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            2
        )

        # ==========================================
        # SAVE SLOT
        # ==========================================
        parking_slots.append({
            "id": slot_id,
            "points": slot.tolist(),
            "center": center.tolist(),
            "length": float(slot_length),
            "width": float(slot_width)
        })

# ==========================================
# SAVE JSON
# ==========================================
with open("parking_map.json", "w") as f:
    json.dump(parking_slots, f, indent=4)

print("Saved: parking_map.json")

# ==========================================
# SAVE IMAGE
# ==========================================
cv2.imwrite("parking_map.jpg", image)

print("Saved: parking_map.jpg")

# ==========================================
# SHOW
# ==========================================
cv2.imshow("Parking Map", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
