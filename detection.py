# Файл определяет занятость размеченных парковочных мест

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

# Минимальное перекрытие
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

        vehicle_polygons.append(
            Polygon(pts)
        )

        # =============================================
        # DRAW VEHICLE
        # =============================================
        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=(0,255,0),
            thickness=2
        )

# =====================================================
# OCCUPANCY DETECTION
# =====================================================

free_count = 0
occupied_count = 0

for slot_data in parking_slots:

    slot_id = slot_data["id"]

    slot_pts = np.array(
        slot_data["points"],
        dtype=np.int32
    )

    slot_polygon = Polygon(slot_pts)

    occupied = False

    # =================================================
    # CHECK OVERLAP
    # =================================================

    for vehicle_polygon in vehicle_polygons:

        intersection = slot_polygon.intersection(vehicle_polygon).area

        slot_area = slot_polygon.area

        overlap_ratio = intersection / slot_area

        if overlap_ratio > OCCUPANCY_THRESHOLD:
            occupied = True
            break

    # =================================================
    # DRAW SLOT
    # =================================================

    if occupied:

        color = (0,0,255)  # RED
        text = f"{slot_id}: OCCUPIED"

        occupied_count += 1

    else:

        color = (0,255,0)  # GREEN
        text = f"{slot_id}: FREE"

        free_count += 1

    cv2.polylines(
        image,
        [slot_pts],
        isClosed=True,
        color=color,
        thickness=2
    )

    cv2.putText(
        image,
        text,
        tuple(slot_pts[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

# =====================================================
# SUMMARY
# =====================================================

summary = f"Free: {free_count}  Occupied: {occupied_count}"

cv2.putText(
    image,
    summary,
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255,255,255),
    3
)

print(summary)

# =====================================================
# SAVE
# =====================================================

cv2.imwrite("occupancy_result.jpg", image)

print("Saved: occupancy_result.jpg")

# =====================================================
# SHOW
# =====================================================

cv2.imshow("Parking Occupancy", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
