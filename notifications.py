import os
import io
import time
import html
import threading
from typing import Optional
import requests
import cv2
from dotenv import load_dotenv

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    gTTS = None  # type: ignore
    HAS_GTTS = False

load_dotenv()

# Cooldown tracking: maps (hazard_type, location) -> last_dispatched_timestamp
_ALERT_COOLDOWNS = {}
_COOLDOWN_LOCK = threading.Lock()
COOLDOWN_SECONDS = 300  # 5 minutes suppression between repeated pings


def _send_telegram_payload(message: str, image_bytes: Optional[bytes] = None, hazard_type: str = "hazard", location: str = "Lab A"):
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip("\"'")
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip("\"'")

    if not token or not chat_id:
        print("[Notifier Error] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
        return

    # 1. Send photo evidence with HTML message summary as caption
    photo_sent = False
    if image_bytes:
        photo_url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {"photo": ("evidence.jpg", image_bytes, "image/jpeg")}
        data = {
            "chat_id": chat_id,
            "caption": message,
            "parse_mode": "HTML"
        }
        try:
            res = requests.post(photo_url, data=data, files=files, timeout=10)
            res.raise_for_status()
            photo_sent = True
        except Exception as e:
            print(f"[Notifier Error] Failed sending Telegram photo: {e}. Falling back to text message.")

    # Fallback to pure text message if photo missing or upload failed
    if not photo_sent:
        msg_url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            res = requests.post(
                msg_url,
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=5
            )
            res.raise_for_status()
        except Exception as e:
            print(f"[Notifier Error] Failed sending Telegram text message: {e}")

    # 2. Synthesize and send voice memo alert note
    try:
        if not HAS_GTTS or gTTS is None:
            print("[Notifier Warning] gTTS library not installed; skipping voice note.")
            return

        clean_hazard = hazard_type.strip().lower()
        if "fire" in clean_hazard or "flame" in clean_hazard:
            spoken_text = f"Fire detected at {location}."
        elif "smoke" in clean_hazard:
            spoken_text = f"Smoke detected at {location}."
        elif "spill" in clean_hazard:
            spoken_text = f"Chemical spill detected at {location}."
        else:
            spoken_text = f"{hazard_type.strip()} detected at {location}."

        tts = gTTS(text=spoken_text, lang="en")
        audio_buf = io.BytesIO()
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)

        voice_url = f"https://api.telegram.org/bot{token}/sendVoice"
        files = {"voice": ("alert.mp3", audio_buf, "audio/mpeg")}
        data = {
            "chat_id": chat_id,
            "caption": f"🔊 Spoken Dispatch: {spoken_text}"
        }
        res_voice = requests.post(voice_url, data=data, files=files, timeout=10)
        res_voice.raise_for_status()
    except Exception as e:
        print(f"[Notifier Error] Failed sending voice memo: {e}")


def dispatch_hazard_alert(hazard_type: str, location: str = "Lab A", frame_bgr=None):
    """
    Non-blocking dispatcher. Invoked from the vision loop or API when an emergency latches.
    Dispatches both text/photo notification and a voice alert memo to Telegram.
    """
    now = time.monotonic()
    cooldown_key = (hazard_type.strip().lower(), location.strip().lower())
    with _COOLDOWN_LOCK:
        last_sent = _ALERT_COOLDOWNS.get(cooldown_key, 0)
        # Suppress notifications if an alert of this type and location was sent recently
        if now - last_sent < COOLDOWN_SECONDS:
            return
        _ALERT_COOLDOWNS[cooldown_key] = now

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    clean_hazard = html.escape(hazard_type.upper())
    clean_location = html.escape(location)

    is_emergency = any(k in hazard_type.lower() for k in ("fire", "smoke", "spill", "flame", "emergency"))
    alert_title = "SAFELABS EMERGENCY ALERT" if is_emergency else "SAFELABS SAFETY INFRACTION"
    severity_label = "CRITICAL" if is_emergency else "HIGH"
    icon = "🚨" if is_emergency else "⚠️"

    message = (
        f"{icon} <b>{alert_title}</b>\n\n"
        f"• <b>Type:</b> <code>{clean_hazard}</code>\n"
        f"• <b>Location:</b> {clean_location}\n"
        f"• <b>Timestamp:</b> <code>{timestamp}</code>\n"
        f"• <b>Severity:</b> <b>{severity_label}</b>\n\n"
        f"⚠️ <i>Immediate verification and laboratory safety response required.</i>"
    )

    # Encode frame in memory to prevent disk writes
    image_bytes = None
    if frame_bgr is not None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            image_bytes = buf.tobytes()

    # Dispatch in a daemon thread so the RTSP/inference loop never stalls
    worker = threading.Thread(
        target=_send_telegram_payload,
        args=(message, image_bytes, hazard_type, location),
        daemon=True
    )
    worker.start()
