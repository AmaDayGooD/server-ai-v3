

import cv2
import numpy as np
from ultralytics import YOLO

# =========================================
# MODEL
# =========================================
model = YOLO("runs/obb/parking_obb/weights/best.pt")

# =========================================
# IMAGE
# =========================================
image_path = "test.jpg"

results = model(image_path)

for result in results:

    image = result.orig_img.copy()

    if result.obb is None:
        continue

    boxes = result.obb.xyxyxyxy.cpu().numpy()

    for box in boxes:

        pts = box.astype(np.float32)

        # =====================================
        # Рисуем автомобиль
        # =====================================
        cv2.polylines(
            image,
            [pts.astype(int)],
            isClosed=True,
            color=(0,255,0),
            thickness=2
        )

        # =====================================
        # Центр
        # =====================================
        center = np.mean(pts, axis=0)

        # =====================================
        # Векторы сторон
        # =====================================
        edge1 = pts[1] - pts[0]
        edge2 = pts[2] - pts[1]

        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)

        # =====================================
        # Определяем длинную сторону
        # =====================================
        if len1 > len2:
            main_vec = edge1 / len1
            side_vec = edge2 / len2
            car_length = len1
            car_width = len2
        else:
            main_vec = edge2 / len2
            side_vec = edge1 / len1
            car_length = len2
            car_width = len1

        # =====================================
        # Размер parking slot
        # =====================================
        slot_length = car_length * 1.3
        slot_width = car_width * 1.5

        # =====================================
        # Строим parking slot
        # =====================================
        p1 = center - main_vec * slot_length/2 - side_vec * slot_width/2
        p2 = center + main_vec * slot_length/2 - side_vec * slot_width/2
        p3 = center + main_vec * slot_length/2 + side_vec * slot_width/2
        p4 = center - main_vec * slot_length/2 + side_vec * slot_width/2

        slot_pts = np.array([p1, p2, p3, p4], dtype=np.int32)

        # =====================================
        # Рисуем parking slot
        # =====================================
        cv2.polylines(
            image,
            [slot_pts],
            isClosed=True,
            color=(255,0,0),
            thickness=2
        )

# =========================================
# SAVE
# =========================================
cv2.imwrite("parking_slots.jpg", image)

print("Saved: parking_slots.jpg")

# =========================================
# SHOW
# =========================================
cv2.imshow("Parking Slots", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
