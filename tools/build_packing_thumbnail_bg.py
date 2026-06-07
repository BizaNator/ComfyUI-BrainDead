#!/usr/bin/env python3
"""
Compose the BD-game_engine_packing thumbnail background (1180x680).

The available shots are all full-workflow captures (the RGB-pack result shows only
as small node previews), so the cleanest card uses the *current* centered workflow
(the 24-node graph WITH the Crop & Center column) as the faded base, and pulls the
small RGB-pack preview out of it as a single bright hero inset on the right.

Sources live under example_workflows/screenshots/. Re-run after new screenshots,
then regen the thumbnail.
"""
import os
from PIL import Image, ImageDraw

W, H = 1180, 680
GREEN = (22, 163, 74)
HERE = os.path.join(os.path.dirname(__file__), "..", "example_workflows", "screenshots")


def cover(img, w, h):
    r = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
    x = (img.width - w) // 2
    return img.crop((x, 0, x + w, h))


# base = the current centered workflow (has the Crop & Center nodes)
after = Image.open(os.path.join(HERE, "game_engine_packing_after.png")).convert("RGB")
base = cover(after, W, H)

# Hero inset: crop the RGB-pack preview from the top-right of the source capture.
# (the Save RGB Pack node preview sits in the upper-right ~ x:0.80-0.92, y:0.16-0.34)
aw, ah = after.size
crop = after.crop((int(aw * 0.795), int(ah * 0.15), int(aw * 0.925), int(ah * 0.36)))
ch_h = 300
ch_w = max(1, int(crop.width * ch_h / crop.height))
crop = crop.resize((ch_w, ch_h), Image.LANCZOS)
hx, hy = W - ch_w - 40, (H - ch_h) // 2
d = ImageDraw.Draw(base)
d.rectangle([hx - 4, hy - 4, hx + ch_w + 4, hy + ch_h + 4], fill=(12, 12, 16))
base.paste(crop, (hx, hy))
d.rectangle([hx - 4, hy - 4, hx + ch_w + 4, hy + ch_h + 4], outline=GREEN, width=4)

out = os.path.join(HERE, "game_engine_packing_bg.png")
base.save(out, quality=95)
print(f"wrote {out} ({base.size[0]}x{base.size[1]}) — hero crop {crop.size}")
