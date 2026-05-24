import json

def get_keyboard():
    keyboard = {
        "one_time": False,
        "buttons": [
            [
                {
                    "action": {
                        "type": "text",
                        "label": "📊 Статус"
                    },
                    "color": "positive"
                }
            ],
            [
                {
                    "action": {
                        "type": "text",
                        "label": "ℹ️ Помощь"
                    },
                    "color": "secondary"
                }
            ]
        ]
    }

    return json.dumps(keyboard, ensure_ascii=False)