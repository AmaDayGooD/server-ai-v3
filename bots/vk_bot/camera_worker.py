import time
from camera import get_frame
import bots.vk_bot.state as state

def camera_worker():
    print("[camera] started")

    while True:
        try:
            frame = get_frame()

            if frame is not None:
                with state.frame_lock:
                    state.latest_frame = frame  # ВАЖНО

        except Exception as e:
            print("[camera error]", e)

        time.sleep(0.5)