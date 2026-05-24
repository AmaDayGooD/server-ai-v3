import random
from vk_api.longpoll import VkEventType

from .config import longpoll, vk
from .handlers import handle_status, handle_help, handle_default

def run_bot():
    print("VK Bot started")

    for event in longpoll.listen():

        if event.type == VkEventType.MESSAGE_NEW and event.to_me:

            text = event.text.lower()

            if text in ["📊 статус", "статус"]:
                handle_status(event)

            elif text in ["ℹ️ помощь", "помощь"]:
                handle_help(event)

            else:
                handle_default(event)