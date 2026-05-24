import cv2
import numpy as np
import json
from ultralytics import YOLO
from shapely.geometry import Polygon

from most_loaded import get_top_k_loaded_images

# ==========================================
model = YOLO("runs/obb/parking_obb/weights/best.pt")

IMAGE_DIR = "parking_photos"

top_images = get_top_k_loaded_images(IMAGE_DIR, k=3)

print("Selected images:")
for img, cnt in top_images:
    print(img, cnt)

# ==========================================
# COLLECT ALL SLOTS FROM MULTIPLE IMAGES
# ==========================================
all_slots = []

for IMAGE_PATH, _ in top_images:

    results = model(IMAGE_PATH)

    for result in results:

        image = result.orig_img.copy()

        if result.obb is None:
            continue

        boxes = result.obb.xyxyxyxy.cpu().numpy()

        for idx, box in enumerate(boxes):

            pts = np.array(box, dtype=np.float32)

            edge1 = pts[1] - pts[0]
            edge2 = pts[2] - pts[1]

            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)

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

            center = np.mean(pts, axis=0)

            slot_length = car_length * 1.15
            slot_width = car_width * 1.2

            p1 = center - long_vec * slot_length/2 - short_vec * slot_width/2
            p2 = center + long_vec * slot_length/2 - short_vec * slot_width/2
            p3 = center + long_vec * slot_length/2 + short_vec * slot_width/2
            p4 = center - long_vec * slot_length/2 + short_vec * slot_width/2

            slot = np.array([p1, p2, p3, p4], dtype=np.int32)

            all_slots.append(slot)

def overlap_ratio(a, b):
    poly_a = Polygon(a)
    poly_b = Polygon(b)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0

    inter = poly_a.intersection(poly_b).area
    min_area = min(poly_a.area, poly_b.area)

    if min_area == 0:
        return 0

    return inter / min_area


unique_slots = []

for slot in all_slots:

    is_duplicate = False

    for u in unique_slots:

        if overlap_ratio(slot, u) >= 0.3:  # 🔥 10% перекрытия
            is_duplicate = True
            break

    if not is_duplicate:
        unique_slots.append(slot)

# ==========================================
# SAVE RESULT
# ==========================================
parking_slots = []

for i, slot in enumerate(unique_slots):
    parking_slots.append({
        "id": i + 1,
        "points": slot.tolist()
    })

with open("parking_map.json", "w") as f:
    json.dump(parking_slots, f, indent=4)

# ==========================================
# VISUALIZATION (OPTIONAL)
# ==========================================
img = cv2.imread(top_images[0][0])

for slot in unique_slots:
    cv2.polylines(img, [slot], True, (255,0,0), 2)

cv2.imwrite("parking_map.jpg", img)


print("Saved improved parking map")