import os
import vk_api
from dotenv import load_dotenv
from vk_api.longpoll import VkLongPoll
from vk_api.upload import VkUpload

load_dotenv()

VK_TOKEN = os.getenv("VK_TOKEN")

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
upload = VkUpload(vk_session)