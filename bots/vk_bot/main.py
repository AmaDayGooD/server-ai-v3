import threading
from bots.vk_bot.bot import run_bot
from bots.vk_bot.camera_worker import camera_worker

threading.Thread(target=camera_worker, daemon=True).start()

print("Camera worker started")

run_bot()