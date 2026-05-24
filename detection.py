import cv2
import json
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "runs/obb/parking_obb/weights/best.pt"
PARKING_MAP_PATH = "parking_map.json"

OCCUPANCY_THRESHOLD = 0.3

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

# ==========================================
# LOAD PARKING MAP
# ==========================================

with open(PARKING_MAP_PATH, "r") as f:
    parking_slots = json.load(f)

# ==========================================
# MAIN FUNCTION
# ==========================================

def detect_parking(image):

    draw_image = image.copy()

    results = model(image)

    vehicle_polygons = []

    # ======================================
    # VEHICLES
    # ======================================

    for result in results:

        if result.obb is None:
            continue

        boxes = result.obb.xyxyxyxy.cpu().numpy()

        for box in boxes:

            pts = np.array(box, dtype=np.int32)

            vehicle_polygons.append(
                Polygon(pts)
            )

    # ======================================
    # PARKING ANALYSIS
    # ======================================

    free_count = 0
    occupied_count = 0

    for slot_data in parking_slots:

        slot_pts = np.array(
            slot_data["points"],
            dtype=np.int32
        )

        slot_polygon = Polygon(slot_pts)

        occupied = False

        for vehicle_polygon in vehicle_polygons:

            intersection = slot_polygon.intersection(
                vehicle_polygon
            ).area

            slot_area = slot_polygon.area

            if slot_area == 0:
                continue

            overlap_ratio = intersection / slot_area

            if overlap_ratio > OCCUPANCY_THRESHOLD:
                occupied = True
                break

        # ==================================
        # DRAW ONLY FREE
        # ==================================

        if not occupied:

            free_count += 1

            cv2.polylines(
                draw_image,
                [slot_pts],
                isClosed=True,
                color=(0,255,0),
                thickness=2
            )

        else:
            occupied_count += 1

    return draw_image, free_count, occupied_count