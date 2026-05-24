# строит постоянную карту парковки на основе НЕСКОЛЬКИХ изображений.
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# =====================================================
# НАСТРОЙКИ
# =====================================================

IMAGE_DIR = "images"
MODEL_PATH = "runs/obb/parking_obb/weights/best.pt"

# Максимальная дистанция между центрами
# одной и той же парковки
DBSCAN_EPS = 30

# Минимум обнаружений для parking slot
MIN_SAMPLES = 2

# Размер parking slot
SLOT_SCALE_LENGTH = 1.15
SLOT_SCALE_WIDTH = 1.3

# =====================================================
# ЗАГРУЗКА МОДЕЛИ
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# ХРАНИЛИЩЕ ВСЕХ ОБНАРУЖЕНИЙ
# =====================================================

all_centers = []
all_boxes = []

# =====================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# =====================================================

image_files = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Images found: {len(image_files)}")

for image_path in image_files:

    print(f"Processing: {image_path}")

    results = model(image_path)

    for result in results:

        if result.obb is None:
            continue

        boxes = result.obb.xyxyxyxy.cpu().numpy()

        for box in boxes:

            pts = np.array(box, dtype=np.float32)

            center = np.mean(pts, axis=0)

            all_centers.append(center)
            all_boxes.append(pts)

# =====================================================
# ПРОВЕРКА
# =====================================================

if len(all_centers) == 0:
    raise Exception("Автомобили не найдены")

all_centers = np.array(all_centers)

print(f"Total detections: {len(all_centers)}")

# =====================================================
# КЛАСТЕРИЗАЦИЯ
# =====================================================

clustering = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=MIN_SAMPLES
).fit(all_centers)

labels = clustering.labels_

unique_labels = set(labels)

print(f"Parking slots found: {len(unique_labels) - (1 if -1 in labels else 0)}")

# =====================================================
# БЕРЁМ ПЕРВОЕ ИЗОБРАЖЕНИЕ КАК ОСНОВУ
# =====================================================

base_image = cv2.imread(image_files[0])

# =====================================================
# СТРОИМ PARKING SLOTS
# =====================================================

for cluster_id in unique_labels:

    # -1 = noise
    if cluster_id == -1:
        continue

    indices = np.where(labels == cluster_id)[0]

    cluster_boxes = [all_boxes[i] for i in indices]

    # =================================================
    # УСРЕДНЯЕМ ЦЕНТРЫ
    # =================================================

    centers = [np.mean(box, axis=0) for box in cluster_boxes]
    center = np.mean(centers, axis=0)

    # =================================================
    # БЕРЁМ ПЕРВЫЙ BOX КАК ЭТАЛОН
    # =================================================

    ref_box = cluster_boxes[0]

    edge1 = ref_box[1] - ref_box[0]
    edge2 = ref_box[2] - ref_box[1]

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

    # =================================================
    # РАЗМЕРЫ PARKING SLOT
    # =================================================

    slot_length = car_length * SLOT_SCALE_LENGTH
    slot_width = car_width * SLOT_SCALE_WIDTH

    # =================================================
    # СТРОИМ SLOT
    # =================================================

    p1 = center - long_vec * slot_length/2 - short_vec * slot_width/2
    p2 = center + long_vec * slot_length/2 - short_vec * slot_width/2
    p3 = center + long_vec * slot_length/2 + short_vec * slot_width/2
    p4 = center - long_vec * slot_length/2 + short_vec * slot_width/2

    slot = np.array([p1, p2, p3, p4], dtype=np.int32)

    # =================================================
    # РИСУЕМ SLOT
    # =================================================

    cv2.polylines(
        base_image,
        [slot],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2
    )

    # =================================================
    # ID SLOT
    # =================================================

    text_pos = tuple(slot[0])

    cv2.putText(
        base_image,
        f"Slot {cluster_id}",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )

# =====================================================
# СОХРАНЕНИЕ
# =====================================================

cv2.imwrite("parking_map.jpg", base_image)

print("Saved: parking_map.jpg")
