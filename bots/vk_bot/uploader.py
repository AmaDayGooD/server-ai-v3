import os
import time
from vk_api.upload import VkUpload
from .config import vk_session


def safe_upload(path):
    for i in range(3):
        try:
            upload = VkUpload(vk_session)
            result = upload.photo_messages(path)
            return result
        except Exception as e:
            print(f"[upload retry {i}]", e)
            time.sleep(1)

    return None


def upload_photo(path):
    result = safe_upload(path)

    if not result:
        return None

    photo = result[0]

    return f"photo{photo['owner_id']}_{photo['id']}"