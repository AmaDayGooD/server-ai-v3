import cv2
import json
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "runs/obb/parking_obb/weights/best.pt"
PARKING_MAP_PATH = "parking_map.json"
IMAGE_PATH = "test2.jpg"

OCCUPANCY_THRESHOLD = 0.3

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# LOAD IMAGE
# =====================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception(f"Cannot load image: {IMAGE_PATH}")

# =====================================================
# LOAD PARKING MAP
# =====================================================

with open(PARKING_MAP_PATH, "r") as f:
    parking_slots = json.load(f)

# =====================================================
# DETECT VEHICLES
# =====================================================

results = model(IMAGE_PATH)

vehicle_polygons = []

for result in results:
    if result.obb is None:
        continue

    boxes = result.obb.xyxyxyxy.cpu().numpy()

    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        vehicle_polygons.append(Polygon(pts))

# =====================================================
# DRAW ONLY FREE PARKING SLOTS
# =====================================================

for slot_data in parking_slots:

    slot_pts = np.array(slot_data["points"], dtype=np.int32)
    slot_polygon = Polygon(slot_pts)

    occupied = False

    slot_area = slot_polygon.area

    for vehicle_polygon in vehicle_polygons:

        intersection = slot_polygon.intersection(vehicle_polygon).area

        if slot_area > 0:
            overlap_ratio = intersection / slot_area
            if overlap_ratio > OCCUPANCY_THRESHOLD:
                occupied = True
                break

    # ❗ РИСУЕМ ТОЛЬКО СВОБОДНЫЕ
    if not occupied:
        cv2.polylines(
            image,
            [slot_pts],
            isClosed=True,
            color=(0, 255, 0),  # GREEN = FREE
            thickness=2
        )

# =====================================================
# SAVE + SHOW
# =====================================================

cv2.imwrite("occupancy_result.jpg", image)

print("Saved: occupancy_result.jpg")

cv2.imshow("Free Parking Slots", image)
cv2.waitKey(0)
cv2.destroyAllWindows()