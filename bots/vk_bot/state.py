import threading

latest_frame = None
frame_lock = threading.Lock()