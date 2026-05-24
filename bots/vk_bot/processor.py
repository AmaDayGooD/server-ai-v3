import cv2
import random

from detection import detect_parking
import bots.vk_bot.state as state
from .uploader import upload_photo
from .config import vk, upload

TEMP_IMAGE = "result.jpg"

def process_status(user_id):
    try:
        with state.frame_lock:
            if state.latest_frame is None:
                raise Exception("camera not ready")
            frame = state.latest_frame.copy()

        result_image, free, occupied = detect_parking(frame)

        cv2.imwrite(TEMP_IMAGE, result_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        attachment = upload_photo(TEMP_IMAGE)

        if not attachment:
            raise Exception("upload failed")

        vk.messages.send(
            user_id=user_id,
            message=f"🅿 Парковка\nСвободно: {free}\nЗанято: {occupied}",
            attachment=attachment,
            random_id=random.randint(1, 999999999)
        )

    except Exception as e:
        vk.messages.send(
            user_id=user_id,
            message=f"Ошибка: {str(e)[:200]}",
            random_id=random.randint(1, 999999999)
        )