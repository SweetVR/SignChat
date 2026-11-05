"""Draws the small in-VR status panel (a Pillow image)."""

from __future__ import annotations

import math
import sys
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

W, H = 768, 208              # drawing size
OUT_W, OUT_H = 576, 156      # size sent to SteamVR (setOverlayRaw has a size limit; keep it well under 1 MB)
BG = (18, 20, 26, 225)
FG = (240, 242, 246, 255)
MUTED = (150, 156, 168, 255)
GREEN = (72, 214, 140, 255)
AMBER = (255, 186, 73, 255)
RED = (255, 99, 99, 255)
TRACK = (60, 64, 76, 255)
SHORT = {"_rest": "Rest", "_space": "Spc", "_backspace": "Del", "_send": "Send", "_clear": "Clr"}


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False):
    names = (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]) if sys.platform.startswith("win") \
        else (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> str:
    """Keep the end of the text (the newest letters) if it doesn't fit."""
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength("…" + text, font=font) > max_w:
        text = text[1:]
    return "…" + text


def render_hud(state: dict) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=28, fill=BG)

    rec = state.get("rec")
    if rec:
        big = SHORT.get(rec["label"], rec["label"])
        phase = rec["phase"]
        accent = AMBER if phase != "capture" else RED
        progress = rec.get("progress", 0.0)
        take = f"  {rec.get('take_no', 1)}/{rec.get('takes_total', 1)}"
        title = ("Next sign" if phase == "prompt" else "Get ready" if phase == "countdown" else "Recording") + take
        if phase == "countdown":
            verb = "Relax your hand" if rec["label"] == "_rest" else "Form the sign"
            line = f"{verb} · {max(1, math.ceil(rec.get('remaining', 0)))}"
        elif phase == "capture":
            line = f"Do the {rec['label']} motion now" if rec.get("kind") == "motion" else (
                "Relax and move your hand naturally" if rec["label"] == "_rest" else "Hold it still")
        else:
            line = "Recording starts soon"
    else:
        enabled = state.get("enabled")
        label = state.get("label")
        big = (label if label and not label.startswith("_") else "·") if enabled else "II"
        accent = GREEN if enabled else MUTED
        progress = state.get("hold", 0.0) if enabled else 0.0
        title = "Signing" if enabled else "Paused"
        if enabled and not state.get("active", True):
            title = "Signing · hand lowered"
        line = state.get("text") or ("Fingerspell to type" if enabled else "Press Start in SignChat")

    # big letter with a progress ring
    cx, cy, r = 104, H // 2, 78
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=TRACK, width=10)
    if progress > 0:
        d.arc((cx - r, cy - r, cx + r, cy + r), start=-90, end=-90 + 360 * min(1.0, progress), fill=accent, width=10)
    size = 92
    f_big = _font(size, bold=True)
    while size > 24 and d.textlength(big, font=f_big) > 124:
        size -= 4
        f_big = _font(size, bold=True)
    d.text((cx, cy), big, font=f_big, fill=FG, anchor="mm")

    x0 = 212
    d.ellipse((x0, 40, x0 + 18, 58), fill=accent)
    d.text((x0 + 30, 49), title, font=_font(30, bold=True), fill=FG, anchor="lm")
    f_line = _font(40)
    d.text((x0, 132), _fit(d, line, f_line, W - x0 - 28), font=f_line, fill=FG, anchor="lm")
    return img


def render_hud_bytes(state: dict) -> bytes:
    return render_hud(state).resize((OUT_W, OUT_H), Image.LANCZOS).tobytes()
