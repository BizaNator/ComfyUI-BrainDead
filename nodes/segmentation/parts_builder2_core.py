"""
BD Parts Builder 2 - the image logic, free of ComfyUI (unit-testable with numpy + OpenCV only).

Qwen Image 2.1 layer decomposition of a character head, proven on eight City of Brains headshots
(2026-09-23, UEFN-430). The ComfyUI nodes in parts_builder2.py run the model edits and call these
functions for everything that is not a model call:

  vocabulary / vote   a CLOSED list of items; a Qwen3-VL true/false vote picks the present ones. 2.1
                      draws any item it is asked for, present or not, so nothing outside the vote is
                      ever requested.
  resolve             each item's "visible" 2.1 edit is trusted for WHERE the item is, not its pixels
                      (white sunglasses came back grey, hidden brows were drawn). A pixel belongs to the
                      faithful edit (fidelity >= 0.5) that covers it and matches the source colour best;
                      the visible layer is the SOURCE's own pixels there, so stacked they rebuild it.
  paint order         where two COMPLETE layers overlap, whichever the source shows is in front.
  back / front        a complete layer is a whole object; its pixels over bare head skin in the source
                      are behind the head (a hat's back brim), the rest in front.
  plate frame         a bald head that fills < 90% of the frame is re-framed to 94% (a hat sets the
                      source framing; su_wee's bald head was 57% of the frame - too few pixels to texture).
  frame proof         coarse-to-fine affine ECC over regions WITH features; a weak direct fit (ecc < 0.7)
                      is proven through the chain (input's fit x this fit).
  reassembly          back parts -> featureless plate (matte) -> front parts, against the source.
"""

import json
import re

import cv2
import numpy as np
from PIL import Image

# ── vocabulary ────────────────────────────────────────────────────────────────
# one line per item:  key | words the vision model is asked about | noun the edit prompt uses | accessory
DEFAULT_VOCABULARY = """hair | hair | hair | no
eyebrows | eyebrows | eyebrows | no
eyes | eyes | eyes | no
mouth | mouth | mouth and lips | no
beard | beard | beard | no
moustache | moustache | moustache | no
glasses | glasses | glasses | yes
sunglasses | sunglasses | sunglasses | yes
hat | hat or cap | hat | yes
helmet | helmet | helmet | yes
headband | headband or bandana | headband or bandana | yes
headphones | headphones | headphones | yes
earrings | earrings | earrings | yes
piercings | piercings | piercings | yes"""

NEVER_RECOLOUR = {"eyes", "mouth"}


def parse_vocabulary(text):
    """-> list of dicts {key, ask, noun, accessory}; blank lines and # comments skipped."""
    out = []
    for line in (text or DEFAULT_VOCABULARY).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = [x.strip() for x in line.split("|")]
        key = f[0].lower()
        ask = f[1] if len(f) > 1 and f[1] else key
        noun = f[2] if len(f) > 2 and f[2] else key
        acc = len(f) > 3 and f[3].lower() in ("yes", "y", "true", "1", "accessory")
        out.append({"key": key, "ask": ask, "noun": noun, "accessory": acc})
    return out


def vlm_prompt(vocab):
    return ("Look at this character's head. For each item in the list below, answer true only if that item is "
            "clearly visible in the image, otherwise false. Do not add any item that is not in the list. List: "
            + ", ".join(v["ask"] for v in vocab) + ". Reply with one JSON object that maps every listed item "
            "to true or false, and nothing else.")


def _json_objects(text):
    """Every JSON object in text, in order (a stray brace in the prose no longer loses the vote)."""
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, n = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                out.append(obj)
            i += max(n, 1)
        except ValueError:
            i += 1


def _truthy(val):
    return bool(val) and str(val).strip().lower() not in ("false", "0", "no", "none", "null")


def parse_vote(text, vocab):
    """VLM reply -> {key: bool}. Reads every JSON object in the reply (fences, prose and stray braces
    tolerated); a string "false" counts as absent. With no JSON object, a typed list works: 'hair, eyes, hat'."""
    text = text or ""
    vote = {v["key"]: False for v in vocab}
    words = {v["key"]: {v["key"], v["ask"].lower()} | {w.strip() for w in re.split(r"\bor\b|/|,", v["ask"].lower()) if w.strip()}
             for v in vocab}
    objs = _json_objects(text)
    for raw in objs:
        for k, val in raw.items():
            kl = str(k).strip().lower()
            for key, ws in words.items():
                if kl in ws:
                    vote[key] = vote[key] or _truthy(val)
    if objs:
        return vote
    for tok in re.split(r"[,\n;]+", text):
        t = tok.strip().lower()
        for key, ws in words.items():
            if t and t in ws:
                vote[key] = True
    return vote


# ── prompts (Qwen Image 2.1 guide form: operation first, one keep clause, affirmative, RGBA form) ─────────
T0 = "This is an RGBA image with transparency. "
T1 = " The image has alpha channel and the background is transparent."


def visible_prompt(noun, colour=""):
    cn = ("%s %s" % (colour, noun)) if colour else noun
    return T0 + ("Extract the %s from the image: keep only the character's %s exactly where it is, at the same size, "
                 "shape, colour and hard-edged low-poly 3D render style, and make everything else - the face, head, "
                 "skin, ears and background - fully transparent." % (noun, cn)) + T1


def complete_prompt(noun, colour=""):
    cn = ("%s %s" % (colour, noun)) if colour else noun
    return T0 + ("Decompose the character's head into separate layers and output only the %s layer: the %s exactly "
                 "where it is in the image, at the same size, shape, colour and hard-edged low-poly 3D render style; "
                 "every other layer is left out and fully transparent." % (noun, cn)) + T1


def base_prompt(remove_nouns, keep_nouns):
    return ("Remove the %s from the character's head and show what was underneath - skin, hair or ears - continuing "
            "the surrounding colours and the hard-edged low-poly facets. Keep the %s, the face, the pose, the head "
            "size and position and the low-poly 3D render style exactly as in the image."
            % (" and the ".join(remove_nouns), ", ".join(keep_nouns) or "rest of the head"))


PLATE_KEEP = ("Keep the rest of the head, the ears, the pose, the head size and position, the lighting and the "
              "low-poly 3D render style exactly as in the image.")
BALD_PROMPT = ("This is an RGBA image with transparency. Make the character's head completely bald: remove the hair on "
               "the head, any hat or head covering, headband or bandana, glasses or sunglasses, headphones and earrings, "
               "the beard and the moustache, and show a smooth, round, complete bald scalp that continues the skin tone "
               "and the hard-edged low-poly facets of the forehead, with clean skin wherever the hair, beard or those "
               "items were. The eyebrows are not part of the hair: keep both eyebrows exactly as they are, whatever "
               "their colour. Keep the face, the eyes, the eyebrows, the lips, the nose and the ears exactly as they "
               "are, and keep the pose, the head size and position and the low-poly 3D render style exactly as in the "
               "image. The image has alpha channel and the background is transparent.")
MOUTHLESS_PROMPT = ("Remove the lips and everything inside the mouth from the character's face: the whole mouth area "
                    "becomes plain skin that continues the surrounding skin tone and the hard-edged low-poly facets, with "
                    "no lip colour, no teeth, no outline and no line, while the chin, the jaw and the cheeks stay exactly "
                    "where they are. Remove the eyebrows the same way, so the brow ridges are plain skin. Keep the eyes "
                    "open and exactly as they are. " + PLATE_KEEP)
EYELESS_PROMPT = ("Remove the eyebrows, the eyes, the lips and everything inside the mouth from the character's face: the "
                  "brow ridges become plain skin, each eye becomes a closed, smooth eyelid of plain skin with no "
                  "eyelashes, no eye line and no eye colour, and the whole mouth area becomes plain skin with no lip "
                  "colour, no teeth, no outline and no line - all of it continuing the surrounding skin tone and the "
                  "hard-edged low-poly facets, while the chin, the jaw and the cheeks stay exactly where they are. "
                  + PLATE_KEEP)
# what each plate stage takes away (the frame proof ignores exactly these regions)
REMOVES = {"bald": {"hair", "beard", "moustache", "__accessories__"},
           "mouthless": {"eyebrows", "mouth"}, "eyeless": {"eyebrows", "mouth", "eyes"}}

COLOURS = {"white": (240, 240, 240), "light grey": (190, 190, 190), "grey": (128, 128, 128), "black": (25, 25, 25),
           "dark brown": (60, 35, 20), "brown": (120, 75, 40), "blonde": (220, 190, 110), "golden yellow": (225, 180, 40),
           "red": (190, 30, 30), "orange": (230, 120, 30), "pink": (235, 140, 170), "purple": (120, 60, 160),
           "dark blue": (30, 45, 120), "blue": (50, 100, 210), "light blue": (130, 180, 230), "green": (50, 140, 60),
           "teal": (40, 140, 140), "silver": (200, 200, 210)}


def colour_name(rgb):
    return min(COLOURS, key=lambda k: sum((a - b) ** 2 for a, b in zip(COLOURS[k], rgb)))


# ── small image helpers ──────────────────────────────────────────────────────
def bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]      # xyxy, exclusive


def over_white(rgba_u8, matte=None):
    """RGBA (or RGB + matte 0..1) composited on pure white -> uint8 RGB."""
    rgb = rgba_u8[..., :3].astype(np.float32)
    a = (rgba_u8[..., 3:4].astype(np.float32) / 255.0) if matte is None else matte[..., None].astype(np.float32)
    return (rgb * a + 255.0 * (1 - a)).clip(0, 255).astype(np.uint8)


def clamp_alpha(alpha01, floor=0.05):
    """2.1's alpha carries faint background noise; zero everything under floor."""
    return alpha01 * (alpha01 > floor)


# ── frame proof ──────────────────────────────────────────────────────────────
def register(ref_rgb, mov_rgb, region, max_scale_err=0.015, max_offset_px=3.0):
    k = max(ref_rgb.shape[:2]) / 1024.0              # kernels and the minimum region were tuned at 1024
    """Affine ECC of mov onto ref on gradient images over `region`, coarse-to-fine (a 20 px shift is 5 px at the
    top level). -> {ecc, scale_x, scale_y, centre_move_px (in 1024 px units), pass}. Identical images pass."""
    if np.array_equal(ref_rgb, mov_rgb):
        return {"identical": True, "pass": True, "ecc": 1.0, "scale_x": 1.0, "scale_y": 1.0, "centre_move_px": 0.0}
    e = int(round(9 * k)) | 1
    m = cv2.erode(region.astype(np.uint8), np.ones((e, e), np.uint8))
    if m.sum() < 2000 * k * k:
        return {"pass": None, "note": "too little region to register"}
    g = lambda a, s: cv2.GaussianBlur(np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)), (0, 0), s)
    ref0 = cv2.cvtColor(np.ascontiguousarray(ref_rgb, np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    mov0 = cv2.cvtColor(np.ascontiguousarray(mov_rgb, np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    w = np.eye(2, 3, dtype=np.float32)
    cc = 0.0
    try:
        for f in (0.25, 0.5, 1.0):
            sz = (int(ref0.shape[1] * f), int(ref0.shape[0] * f))
            ref = cv2.resize(ref0, sz, interpolation=cv2.INTER_AREA)
            mov = cv2.resize(mov0, sz, interpolation=cv2.INTER_AREA)
            mm = cv2.resize(m, sz, interpolation=cv2.INTER_NEAREST)
            if f > 0.25:
                w[:, 2] *= 2.0
            cc, w = cv2.findTransformECC(g(ref, 2), g(mov, 2), w, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), mm * 255, 5)
    except cv2.error as e:
        return {"pass": False, "note": "ECC failed: %s" % str(e)[:80]}
    ys, xs = np.where(m)
    c = np.array([xs.mean(), ys.mean(), 1.0])
    mv = w @ c - c[:2]
    r = {"ecc": round(float(cc), 3), "scale_x": round(float(np.hypot(w[0, 0], w[1, 0])), 4),
         "scale_y": round(float(np.hypot(w[0, 1], w[1, 1])), 4),
         "centre_move_px": round(float(np.hypot(*mv)) * 1024.0 / ref0.shape[1], 2)}
    r["pass"] = bool(abs(r["scale_x"] - 1) <= max_scale_err and abs(r["scale_y"] - 1) <= max_scale_err
                     and r["centre_move_px"] <= max_offset_px)
    return r


def unmeasured(*proofs):
    """True when some frame proof could not be measured at all (ECC did not converge, or too little region) and
    none of them MEASURED a miss - a scale / offset outside the tolerance is a real miss, never a reason to re-try."""
    um = [p.get("pass") is None or str(p.get("note", "")).startswith("ECC failed") for p in proofs]
    return any(um) and not any(p.get("pass") is False and not u for p, u in zip(proofs, um))


def chain_proof(upstream, this, max_scale_err=0.015, max_offset_px=3.0):
    """Compose the input's proof vs the source with this stage's proof vs its input. An upstream stage that was
    not proven (no numbers) proves nothing downstream."""
    need = ("scale_x", "scale_y", "centre_move_px")
    if upstream.get("pass", True) is not True or any(k not in upstream for k in need) or any(k not in this for k in need):
        return {"pass": False, "note": "upstream stage not proven"}
    ch = {"scale_x": round(upstream["scale_x"] * this["scale_x"], 4), "scale_y": round(upstream["scale_y"] * this["scale_y"], 4),
          "centre_move_px": round(upstream["centre_move_px"] + this["centre_move_px"], 2)}
    ch["pass"] = bool(abs(ch["scale_x"] - 1) <= max_scale_err and abs(ch["scale_y"] - 1) <= max_scale_err
                      and ch["centre_move_px"] <= max_offset_px)
    return ch


# ── resolve: visible layers from the SOURCE's pixels ─────────────────────────
def resolve(src_rgb, head, edits, fid_ok=0.5, explained=60.0):
    """src_rgb (H,W,3) uint8; head (H,W) bool (source alpha); edits {tag: RGBA uint8 visible 2.1 edit}.

    -> dict with per-tag {visible RGBA, fidelity, hidden, cover masks}, owner map (0 = none, k+1 = tags[k]),
    head (repaired: an item an edit places on the head is head, whatever the source alpha says)."""
    tags = list(edits)
    src = src_rgb.astype(np.float32)
    if not tags:
        return {"tags": [], "owner": np.zeros(head.shape, np.uint8), "owned_any": np.zeros(head.shape, bool),
                "head": head, "unexplained": 0.0, "parts": {}}
    E = [edits[t].astype(np.float32) for t in tags]
    cover = np.stack([e[..., 3] > 127 for e in E])
    head_r = head | cover.any(0)
    cover &= head_r
    diff = np.stack([np.abs(e[..., :3] - src).mean(-1) for e in E])
    fid = [float((diff[k][cover[k]] < 40).mean()) if cover[k].any() else 0.0 for k in range(len(tags))]
    srcok = np.array([f >= fid_ok for f in fid])[:, None, None]
    cand = cover & srcok & (diff < explained)
    owner = np.where(cand, diff, np.inf).argmin(0)
    owned_any = cand.any(0)
    parts = {}
    for k, t in enumerate(tags):
        own = owned_any & (owner == k)
        vis = np.dstack([src_rgb.astype(np.uint8), (own * 255).astype(np.uint8)])
        ec = cover[k]
        parts[t] = {"visible": vis, "own": own, "edit_cover": ec, "fidelity": round(fid[k], 3),
                    "hidden_or_restyled": fid[k] < fid_ok,
                    "edit_colour": colour_name(tuple(int(c) for c in np.median(src_rgb[ec], 0))) if ec.any() else None,
                    "colour_of_item": colour_name(tuple(int(c) for c in np.median(src_rgb[own], 0))) if own.any() else None}
    omap = np.where(owned_any, owner + 1, 0).astype(np.uint8)
    return {"tags": tags, "owner": omap, "owned_any": owned_any, "head": head_r,
            "unexplained": float((cover.any(0) & ~owned_any).sum() / max(head_r.sum(), 1)), "parts": parts}


def complete_colour_shift(complete_rgba, src_rgb, own):
    both = own & (complete_rgba[..., 3] > 127)
    if both.sum() <= 50:
        return None
    return round(float(np.abs(np.median(complete_rgba[..., :3][both].astype(np.float32), 0)
                              - np.median(src_rgb[both].astype(np.float32), 0)).max()), 1)


def paint_order(tags, complete, owner, depth_far=None, min_px=30):
    """Back -> front. complete {tag: RGBA}; owner map from resolve (k+1 = tags[k]); depth_far {tag: larger = farther}
    (the node passes 1 - Lotus2 median, since Lotus2's map/raw_linear is brighter = nearer; hidden items 2.0).
    A is in front of B where, on their complete-layer overlap, the source shows A (owns more pixels)."""
    C = {t: complete[t][..., 3] > 127 for t in tags}
    front = {t: set() for t in tags}
    evidence = {}
    for i, a in enumerate(tags):
        for j in range(i + 1, len(tags)):
            b = tags[j]
            ov = C[a] & C[b] & (owner > 0)
            na, nb = int((ov & (owner == i + 1)).sum()), int((ov & (owner == j + 1)).sum())
            if max(na, nb) < min_px:
                continue
            f, bk = (a, b) if na >= nb else (b, a)
            front[f].add(bk)
            evidence["%s>%s" % (f, bk)] = [max(na, nb), min(na, nb)]
    far = {t: (depth_far or {}).get(t, 0.5) for t in tags}
    order, left = [], set(tags)
    while left:
        ready = [t for t in left if not (front[t] & left)] or list(left)      # cycle: fall back to depth
        t = max(ready, key=lambda t: (far[t], -tags.index(t)))                # farthest first
        order.append(t)
        left.discard(t)
    return order, evidence


HEADWEAR = ("hat", "helmet", "headband", "headphones", "glasses", "sunglasses")


def headwear_over_hair(order):
    """Head-worn items paint in front of hair. The overlap vote in paint_order flips on near ties (cast run
    2026-09-24: brick_callahan's beanie, hat 42k px vs hair 47k px, went under the complete hair, which covers it).
    Safe both ways because visible_wins clips a front part wherever a LOWER part is what the source shows.
    -> (order, moved)."""
    order, moved = list(order), []
    if "hair" in order:
        for hw in HEADWEAR:
            if hw in order and order.index(hw) < order.index("hair"):
                order.remove(hw)
                order.insert(order.index("hair") + 1, hw)
                moved.append(hw)
    return order, moved


def visible_wins(front_rgba, own, lower_own, src_rgb, fringe=2):
    """A FRONT part reproduces the source wherever the source shows it or something below it:
      * its own visible pixels (own) take the source colours at full alpha - a visible item is always in its layer
        (cast run: boomboom's complete "mouth" came back as giant lips + eyeliner, her real lips were in no layer);
      * it is cleared where a part LOWER in the paint order is visible (lower_own, grown by `fringe` px for the
        antialiased edge) and it is not.
    -> (RGBA, {"visible_added_px", "clipped_px"})."""
    F = front_rgba.copy()
    added = int((own & (F[..., 3] <= 127)).sum())
    F[own, :3] = src_rgb[own]
    F[own, 3] = 255
    below = np.zeros(own.shape, bool)
    for m in lower_own:
        below |= m
    if fringe > 0 and below.any():
        below = cv2.dilate(below.astype(np.uint8), np.ones((2 * fringe + 1, 2 * fringe + 1), np.uint8)).astype(bool)
    clip = below & ~own & (F[..., 3] > 0)
    F[clip, 3] = 0
    return F, {"visible_added_px": added, "clipped_px": int(clip.sum())}


def split_back_front(complete_rgba, skin0):
    """Pixels of a complete layer over bare head skin in the source are BEHIND the head."""
    back = (complete_rgba[..., 3] > 127) & skin0
    B, F = complete_rgba.copy(), complete_rgba.copy()
    B[..., 3] = np.where(back, complete_rgba[..., 3], 0)
    F[..., 3] = np.where(back, 0, complete_rgba[..., 3])
    return B, F, int(back.sum())


# ── plate frame ──────────────────────────────────────────────────────────────
def plate_box(matte01, fill_min=0.90, fill_to=0.94):
    """Square source-frame box [x0, y0, side] that makes the head fill `fill_to`; None if it already fills
    `fill_min` of the frame (or the matte is empty)."""
    ys, xs = np.where(matte01 > 0.5)
    if not len(xs):
        return None
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    if max(w, h) >= fill_min * max(matte01.shape[:2]):
        return None
    side = int(round(max(w, h) / fill_to))
    cx, cy = (xs.min() + xs.max() + 1) / 2.0, (ys.min() + ys.max() + 1) / 2.0     # pixel-extent centre
    return [int(np.floor(cx - side / 2.0 + 0.5)), int(np.floor(cy - side / 2.0 + 0.5)), side]


def to_frame(img, box, size, nearest=False, fill=255):
    """A source-frame image cropped to box (outside: white for RGB, transparent for RGBA, False for masks) and
    resized to size x size. box None + size == image size -> unchanged."""
    H, W = img.shape[:2]
    if box is None:
        if (H, W) == (size, size):
            return img
        box = [0, 0, max(H, W)]
    x0, y0, side = box
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    widths = ((pad, pad), (pad, pad)) + (((0, 0),) if img.ndim == 3 else ())
    cv = False if img.dtype == bool else (fill if img.ndim == 2 or img.shape[2] == 3 else 0)
    a = np.pad(img, widths, constant_values=cv)[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if a.dtype == bool:
        return np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.NEAREST if nearest else Image.LANCZOS))


# ── comparison (the red-zone difference) ──────────────────────────────────────
def compare(ref_rgb, cand_rgb, region, gain=3.0, threshold=40):
    """How far a candidate lands from the reference, per pixel. -> (heat uint8 RGB, score dict).

    heat: red = |reference - candidate| (mean over RGB) x gain; the region tinted teal so the outline shows.
    score: mean difference on the region (0 = identical, in 0..255 levels) and the share of the region more
    than `threshold` levels off - the numbers the head chain was judged by (UEFN-430)."""
    d = np.abs(cand_rgb[..., :3].astype(np.float32) - ref_rgb[..., :3].astype(np.float32)).mean(-1)
    heat = np.zeros(ref_rgb.shape[:2] + (3,), np.uint8)
    heat[..., 0] = np.clip(d * gain, 0, 255).astype(np.uint8)
    heat[..., 1] = heat[..., 2] = (region * 40).astype(np.uint8)
    n = int(region.sum())
    score = {"diff_on_region": round(float(d[region].mean()), 2) if n else None,
             "share_over_%d" % threshold: round(float((d[region] > threshold).mean()), 4) if n else None,
             "region_px": n, "max_diff": round(float(d[region].max()), 1) if n else None}
    return heat, score, d


def side_by_side(ref_rgb, cand_rgb, heat, labels=("reference", "candidate", "difference (red = off)"), footer=""):
    """reference | candidate | difference, labelled, one panel."""
    from PIL import ImageDraw
    H, W = ref_rgb.shape[:2]
    bar = max(24, H // 32)
    panel = Image.new("RGB", (W * 3, H + bar * (2 if footer else 1)), "white")
    for i, a in enumerate((ref_rgb[..., :3], cand_rgb[..., :3], heat)):
        panel.paste(Image.fromarray(np.ascontiguousarray(a)), (i * W, bar))
    dr = ImageDraw.Draw(panel)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=max(14, bar - 8))
    except Exception:
        font = None
    for i, t in enumerate(labels):
        dr.text((i * W + 8, 4), t, fill=(0, 0, 0), font=font)
    if footer:
        dr.text((8, H + bar + 4), footer, fill=(0, 0, 0), font=font)
    return np.asarray(panel)


# ── flat eye section + even light (FaceMaker inputs) ────────────────────────
def flat_fill(img_rgb, zone, head=None, grow=3, ring=(8, 22), feather=2.0, min_px=200):
    """Fill each connected zone with ONE flat tone: the median of a skin ring around it. -> (image, tones).

    FaceMaker's engine eye sits on a flat eye section; v26 made it with its "Crop and Fill Eyes" pre-face step,
    which v27 removed. The eyeless plate carries closed lids instead, so the Plates node flattens the MediaPipe
    eye zones itself. The zone is grown by `grow` px to cover the lash line; the ring (`ring` px outside the grown
    zone, inside `head`) is far enough out to miss the lid-crease shadow; the edge is feathered by `feather`."""
    zone = np.asarray(zone).astype(bool)
    out = np.asarray(img_rgb)[..., :3].astype(np.float32).copy()
    head = np.ones(zone.shape, bool) if head is None else np.asarray(head).astype(bool)
    n, lab = cv2.connectedComponents(zone.astype(np.uint8))
    tones = []
    for i in range(1, n):
        z = lab == i
        if z.sum() < min_px:
            continue
        k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        g = cv2.dilate(z.astype(np.uint8), k(grow)) > 0 if grow else z
        rg = (cv2.dilate(g.astype(np.uint8), k(ring[1])) > 0) & ~(cv2.dilate(g.astype(np.uint8), k(ring[0])) > 0) & head
        if rg.sum() < 20:
            continue
        tone = np.median(out[rg], axis=0)
        soft = cv2.GaussianBlur(g.astype(np.float32), (0, 0), feather) if feather else g.astype(np.float32)
        out = out * (1 - soft[..., None]) + tone[None, None, :] * soft[..., None]
        tones.append({"px": int(z.sum()), "tone": [int(round(t)) for t in tone]})
    return np.clip(out, 0, 255).astype(np.uint8), tones


def even_light(img_rgb, head, keep_light=0.5, facet_gain=0.9, sigma=60.0):
    """Even a head's lighting without flattening its facets. -> (image, stats).

    light  = the head's luminance, blurred with a mask-normalised Gaussian of `sigma` px (at 1024; scaled with size)
    facets = luminance - light
    out    = mean + keep_light * (light - mean) + facet_gain * facets, colour carried by ratio; outside `head`
             the image is unchanged. Qwen Image 2.1 reads "cell shaded with dynamic shadowing" as one hard key
             light (a bright side and a shadow side); prompts that ask for even light also flatten the facets.
             This keeps the facets and takes most of the light out. keep_light 0.5 / facet_gain 0.9 matched
             don_juan's v21 albedo (light std 18 vs 23, facet std 42 vs 39, left-right 25 vs 21 levels)."""
    rgb = np.asarray(img_rgb)[..., :3].astype(np.float32)
    H, W = rgb.shape[:2]
    m = np.asarray(head).astype(bool)
    if m.sum() < 100:
        return rgb.astype(np.uint8), {"skipped": "empty head mask"}
    Y = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    mean = float(Y[m].mean())
    sg = sigma * W / 1024.0
    num = cv2.GaussianBlur(np.where(m, Y, 0).astype(np.float32), (0, 0), sg)
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sg)
    light = np.where(den > 1e-3, num / np.maximum(den, 1e-3), mean)
    facets = Y - light
    Yn = mean + keep_light * (light - mean) + facet_gain * facets
    ratio = np.clip(Yn, 0, 255) / np.maximum(Y, 1.0)
    out = np.where(m[..., None], np.clip(rgb * ratio[..., None], 0, 255), rgb)
    stats = {"mean": round(mean, 1), "light_std_in": round(float(light[m].std()), 1),
             "light_std_out": round(float((keep_light * (light - mean))[m].std()), 1),
             "facet_std_in": round(float(facets[m].std()), 1),
             "facet_std_out": round(float((facet_gain * facets)[m].std()), 1)}
    return out.astype(np.uint8), stats


# ── reassembly ───────────────────────────────────────────────────────────────
def reassemble(plate_rgb, matte01, layers_back_front, order, src_rgb, head, owner=None, tags=None):
    """back parts (order) -> plate over matte -> front parts (order). -> (composite uint8, heat uint8, report)."""
    comp = np.full(plate_rgb.shape, 255.0, np.float32)

    def over(c, L):
        a = L[..., 3:4].astype(np.float32) / 255.0
        return c * (1 - a) + L[..., :3].astype(np.float32) * a
    for t in order:
        comp = over(comp, layers_back_front[t][0])
    m = matte01[..., None].astype(np.float32)
    comp = comp * (1 - m) + plate_rgb.astype(np.float32) * m
    for t in order:
        comp = over(comp, layers_back_front[t][1])
    comp = comp.clip(0, 255)
    heat, sc, d = compare(src_rgb, comp, head)
    r = {"diff_on_head": sc["diff_on_region"], "share_of_head_over_40": sc["share_over_40"], "per_layer": {}}
    if owner is not None and tags:
        for k, t in enumerate(tags):
            mm = owner == k + 1
            if mm.any():
                r["per_layer"][t] = round(float(d[mm].mean()), 1)
        skin = head & (owner == 0)
        if skin.any():
            r["per_layer"]["skin"] = round(float(d[skin].mean()), 1)
    return comp.astype(np.uint8), heat, r


# ── PARTS_BUNDLE helpers ─────────────────────────────────────────────────────
def part_info(tag, full_rgba, depth_median, depth_map=None, **extra):
    """A PARTS_BUNDLE entry from a full-frame RGBA layer: cropped to its alpha, xyxy in frame px. None if empty."""
    bb = bbox(full_rgba[..., 3] > 0)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    info = {"img": np.ascontiguousarray(full_rgba[y1:y2, x1:x2]), "xyxy": [x1, y1, x2, y2], "tag": tag,
            "depth_median": float(depth_median)}
    if depth_map is not None:
        info["depth"] = np.ascontiguousarray((np.clip(depth_map[y1:y2, x1:x2], 0, 1) * 255).astype(np.uint8))
    info.update(extra)
    return info


def order_depths(order):
    """Paint order (back -> front) as depth_median values BD_PartsExport / BD_PartsCompose sort by
    (larger = farther): first painted gets the largest."""
    n = max(len(order), 1)
    return {t: round(1.0 - (i + 0.5) / n, 4) for i, t in enumerate(order)}
