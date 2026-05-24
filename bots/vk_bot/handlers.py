import random
from .config import vk
from .processor import process_status
from bots.vk_bot.keyboard import get_keyboard

def handle_status(event):
    vk.messages.send(
        user_id=event.user_id,
        message="⏳ Анализ парковки...",
        random_id=random.randint(1, 999999999),
        keyboard=get_keyboard()
    )

    import threading
    threading.Thread(
        target=process_status,
        args=(event.user_id,),
        daemon=True
    ).start()


def handle_help(event):
    vk.messages.send(
        user_id=event.user_id,
        message="🚗 Parking AI Bot\n\n📊 Статус\nℹ️ Помощь",
        random_id=random.randint(1, 999999999),
        keyboard=get_keyboard()
    )


def handle_default(event):
    vk.messages.send(
        user_id=event.user_id,
        message="🚗 Используй меню",
        random_id=random.randint(1, 999999999),
        keyboard=get_keyboard()
    )