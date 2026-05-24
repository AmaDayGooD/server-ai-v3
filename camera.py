import asyncio
import base64
import cv2
import numpy as np
from playwright.async_api import async_playwright

# ==========================================
# CONFIG
# ==========================================

PAGE_URL = "https://rtsp.ru/embed/GRTfFsQ6/"
VIDEO_SELECTOR = "#video"

# ==========================================
# GLOBAL STATE
# ==========================================

_playwright = None
_browser = None
_context = None
_page = None
_camera_ready = False

_init_lock = asyncio.Lock()
_frame_lock = asyncio.Lock()

# ОДИН event loop на всё приложение
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

# ==========================================
# INIT BROWSER
# ==========================================

async def init_browser():
    global _playwright, _browser, _context, _page

    async with _init_lock:

        if _page is not None and not _page.is_closed():
            return

        await cleanup()

        try:
            _playwright = await async_playwright().start()

            _browser = await _playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            _context = await _browser.new_context(
                viewport={"width": 1280, "height": 720}
            )

            _page = await _context.new_page()

            await _page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=30000)
            await _page.wait_for_selector(VIDEO_SELECTOR, timeout=20000)

            await _page.evaluate(f"""
                async () => {{
                    const video = document.querySelector('{VIDEO_SELECTOR}');
                    if (video) {{
                        video.muted = true;
                        video.playsInline = true;
                        try {{
                            await video.play();
                        }} catch(e) {{}}
                    }}
                }}
            """)

            await asyncio.sleep(2)

            global _camera_ready
            _camera_ready = True

            print("[camera] browser initialized")

        except Exception as e:
            print("[camera] init error:", e)
            await cleanup()
            raise

# ==========================================
# CLEANUP
# ==========================================

async def cleanup():
    global _playwright, _browser, _context, _page, _camera_ready
    _camera_ready = False
    
    try:
        if _page:
            await _page.close()
    except:
        pass

    try:
        if _context:
            await _context.close()
    except:
        pass

    try:
        if _browser:
            await _browser.close()
    except:
        pass

    try:
        if _playwright:
            await _playwright.stop()
    except:
        pass

    _playwright = None
    _browser = None
    _context = None
    _page = None

# ==========================================
# ASYNC FRAME
# ==========================================

async def get_frame_async():
    global _page

    async with _frame_lock:

        if _page is None or _page.is_closed():
            await init_browser()

        try:
            data = await _page.evaluate(f"""
                () => {{
                    const video = document.querySelector('{VIDEO_SELECTOR}');
                    if (!video || video.videoWidth === 0 || video.readyState < 2)
                        return null;

                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;

                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0);

                    return canvas.toDataURL('image/jpeg', 0.85);
                }}
            """)

            if not data:
                return None

            img_bytes = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            return frame

        except Exception as e:
            print("[camera] frame error:", e)
            return None

# ==========================================
# SYNC WRAPPER (VK SAFE)
# ==========================================

def get_frame():
    try:
        return _loop.run_until_complete(get_frame_async())
    except Exception as e:
        print("[camera] sync error:", e)
        return None