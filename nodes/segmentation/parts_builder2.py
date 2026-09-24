"""
BD Parts Builder 2 - Qwen Image 2.1 layer decomposition of a character head (UEFN-430).

Four nodes, built from the head chain proven on eight City of Brains headshots on 2026-09-23:

  BD_PartsVocabulary        the closed item list + the matching Qwen3-VL true/false prompt
  BD_PartsBuilder2          vote -> per item a VISIBLE edit (where) + COMPLETE edit (whole object),
                            visible pixels resolved from the source, paint order, back/front split,
                            head without items  ->  parts / complete_parts (PARTS_BUNDLE)
  BD_PartsBuilder2Plates    bald (RGBA, its alpha = the matte) -> mouthless and eyeless (one edit of the
                            bald each), every stage cut by the matte onto white and frame-proven; small heads
                            re-framed; 1024 or 2048 working size
  BD_PartsBuilder2Assemble  the engine stack: back parts -> eyeless plate -> front parts, against the source
                            -> engine_parts (PARTS_BUNDLE)

Export with the existing BD_PartsExport (PSD / PSB / manifest, sorted by depth_median = paint order).
The image logic lives in parts_builder2_core.py (no ComfyUI imports, unit-tested).

Every edit is the core TextEncodeQwenImage21 (resolution 0 = the reference's own size, so edits are
frame-true) -> KSampler (40 steps, cfg 1, euler / simple) -> VAEDecode (4 channels: 2.1's VAE is RGBA).
Wire the MODEL through QwenImage21Cache as the template does.

Rules the chain enforces (each cost a run to learn):
  * never feed a stage 2.1's raw output - it amplifies input noise (background std 0.9 -> 5.4 -> 48.8);
    every plate stage is cut by the matte onto pure white first
  * at most two generations from the source - a third turns skin to "crinkled foil"
  * never ask for an item the vote did not find - 2.1 draws it
  * no "head only, no neck" wording where the frame must hold - it recomposes
  * a stage that fails its frame proof stops the chain; nothing is built from a rejected plate
"""

import contextlib
import json

import numpy as np
import torch

import comfy.model_management
import comfy.samplers as _samplers
import comfy.utils
from comfy_api.latest import io
from nodes import common_ksampler

from . import parts_builder2_core as C
from .parts_types import PARTS_BUNDLE, ensure_bundle

try:                                            # core node, ComfyUI >= 6bfaacc6 (Qwen Image 2.1)
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21 as _TE21
except Exception:                               # pragma: no cover - reported at run time
    _TE21 = None

CATEGORY = "🧠BrainDead/Segmentation"


@contextlib.contextmanager
def _no_cudnn_attention():
    """Take cuDNN out of ComfyUI's SDPA backend priority for the duration (comfy.ops reads the list per call).

    Qwen Image 2.1's prefix attention (QwenImage21Cache) can fail with "cuDNN Frontend error: No valid execution
    plans built" on its first use in a process that also holds Lotus2's FLUX and Qwen3-VL - seen three times on
    2026-09-23, never with the models already warm - and a retry after that failure aborts the process. So every
    Parts Builder 2 edit runs without the cuDNN backend; memory-efficient attention takes the same mask."""
    try:
        import comfy.ops as _ops
        from torch.nn.attention import SDPBackend
        prio = getattr(_ops, "SDPA_BACKEND_PRIORITY", None)
    except Exception:
        prio = None
    saved = list(prio) if prio is not None else None
    try:
        if prio is not None:
            prio[:] = [b for b in prio if b != SDPBackend.CUDNN_ATTENTION]
        yield
    finally:
        if prio is not None:
            prio[:] = saved


# ── tensors <-> numpy ────────────────────────────────────────────────────────
def _u8(x):
    """float 0..1 array -> uint8 the way SaveImage does it (truncate), so node outputs match saved PNGs."""
    return np.clip(255.0 * x, 0, 255).astype(np.uint8)


def _img_u8(image):
    """IMAGE tensor (B,H,W,C) -> first frame uint8 (H,W,C)."""
    return _u8(image[0].detach().cpu().float().numpy())


def _mask_bool(mask, shape):
    if mask is None:
        return None
    m = mask[0] if mask.ndim == 3 else mask
    m = m.detach().cpu().float().numpy()
    if m.shape != tuple(shape):
        m = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(m)[None, None], size=shape, mode="nearest")[0, 0])
    return m > 0.5


def _t(arr_u8):
    """uint8 (H,W,C) or list of them -> IMAGE tensor."""
    arrs = arr_u8 if isinstance(arr_u8, list) else [arr_u8]
    return torch.from_numpy(np.stack([a.astype(np.float32) / 255.0 for a in arrs]))


def _m(mask01):
    return torch.from_numpy(np.asarray(mask01, np.float32))[None]


def _source(image, head_mask):
    """Source RGB exactly as given (the proven chain fed MASTER 1's RGB to every edit unchanged) + head region.
    head_mask only says where the head is; without it, the head is the non-white pixels."""
    rgb = _img_u8(image)[..., :3]
    head = _mask_bool(head_mask, rgb.shape[:2])
    if head is None:
        head = (rgb.astype(np.int16).min(-1) < 245)
        head = np.asarray(C.cv2.morphologyEx(head.astype(np.uint8), C.cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), bool)
    return np.ascontiguousarray(rgb), head


# The BD_FaceSocketInfill settings FaceMaker v10-v26 built its eye / brow / mouth exclusion mask with (#8580 ->
# #8447): eyes iris-mode inset 2, brow band 12, lip band 6, every zone +6 px, feathered 3, nose off. The lips were a
# "plane" - a box drawn for the old Qwen lip-stamp prompt, far larger than the lips. The exclusion mask only has to
# cover the lips, so the lip zone is a Plates input (lip_zone) and defaults to "organic": the MediaPipe outer lip
# contour with the same +6 px margin as the eyes and brows (owner, 2026-09-23).
FACEMAKER_SOCKETS = dict(face_data_path="", detection_confidence=0.25, eyes=True, brows=True, lips=True, nose=False,
                         oval_subtract_sockets=False, eye_mode="iris", eye_inset=2, expand_x=6, expand_y=6,
                         eyes_expand_x=-1, eyes_expand_y=-1, brows_expand_x=-1, brows_expand_y=-1, brow_band=12,
                         lip_band=6, lips_expand_x=-1, lips_expand_y=-1, nose_expand_x=-1, nose_expand_y=-1, feather=3,
                         eyes_feather=-1, brows_feather=-1, lips_feather=-1, nose_feather=-1, fill_mode="flat",
                         fill_from_guide=False, fill_r=0, fill_g=0, fill_b=0, lip_mode="organic", surround_style="solid",
                         lip_trapezoid_taper=0.7, solid_bevel=False, solid_bevel_width=8, solid_bevel_strength=0.5,
                         solid_bevel_zones="lips", solid_lift_shadow=0.3)


def _feature_masks(bald_u8, lip_zone="organic"):
    """MediaPipe on the bald plate (bald, not faceless: eyes, brows and mouth are still there) ->
    socket_mask (float HxW, the FaceMaker exclusion mask) and feature_mask (RGB: R = mouth / lips, G = eyes,
    B = brows), in the plate frame. Owner, 2026-09-23: detect once when the head is bald but not faceless,
    save the mask with the masters, load it into FaceMaker."""
    from .face_socket_infill import BD_FaceSocketInfill
    H, W = bald_u8.shape[:2]
    out = BD_FaceSocketInfill.execute(image=_t(bald_u8[..., :3]), **dict(FACEMAKER_SOCKETS, lip_mode=lip_zone)).args
    sock, eyes, brows, lips, status = out[2], out[5], out[8], out[9], out[-1]
    f = lambda m: (m[0] if m.ndim == 3 else m).detach().cpu().float().numpy() if m is not None else np.zeros((H, W), np.float32)
    socket = np.clip(f(sock), 0, 1)
    rgb = np.dstack([_u8(np.clip(f(lips), 0, 1)), _u8(np.clip(f(eyes), 0, 1)), _u8(np.clip(f(brows), 0, 1))])
    return socket, rgb, str(status)


class _Q21:
    """One Qwen Image 2.1 edit, exactly the graph the head chain was proven with."""

    def __init__(self, model, clip, vae, steps, cfg, sampler_name, scheduler):
        if _TE21 is None:
            raise RuntimeError("TextEncodeQwenImage21 not found - this ComfyUI core is older than Qwen Image 2.1 "
                               "(needs >= 6bfaacc6). Update ComfyUI.")
        self.model, self.clip, self.vae = model, clip, vae
        self.steps, self.cfg, self.sampler, self.scheduler = steps, cfg, sampler_name, scheduler
        self.count = 0

    def _run(self, img, prompt, seed):
        with torch.inference_mode():                     # as BD_PartsBatchEdit samples
            pos, neg, latent = _TE21.execute(self.clip, prompt, "", vae=self.vae, resolution=0,
                                             images={"image_1": img}).args
            out = common_ksampler(self.model, int(seed), self.steps, self.cfg, self.sampler, self.scheduler,
                                  pos, neg, latent, denoise=1.0)[0]
            return self.vae.decode(out["samples"])

    def __call__(self, image_u8, prompt, seed):
        comfy.model_management.throw_exception_if_processing_interrupted()
        H, W = image_u8.shape[:2]
        img = torch.from_numpy(image_u8.astype(np.float32) / 255.0)[None]
        # Never let cuDNN attention try: when it fails ("No valid execution plans built") the exception leaves
        # ComfyUI's aimdo allocator with pinned pages, and re-entering the model then aborts the whole process
        # (core dump on production, 2026-09-23 20:32). Memory-efficient attention takes the same mask.
        with _no_cudnn_attention():
            dec = self._run(img, prompt, seed)
        if dec.ndim == 5:
            dec = dec.reshape(-1, dec.shape[-3], dec.shape[-2], dec.shape[-1])
        arr = _u8(dec[0].detach().cpu().float().numpy())
        if arr.shape[-1] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
        if arr.shape[:2] != (H, W):              # resolution 0 rounds each side to a multiple of 32
            arr = np.asarray(C.Image.fromarray(arr).resize((W, H), C.Image.LANCZOS))
        self.count += 1
        return arr


def _sampler_inputs(seed_default=20260931):
    return [
        io.Int.Input("seed", default=seed_default, min=0, max=0xffffffffffffffff, control_after_generate=True,
                     tooltip="Every edit uses this seed; a frame-proof retry uses seed+1, seed+2 ..."),
        io.Int.Input("steps", default=40, min=1, max=200, tooltip="Qwen Image 2.1 model card: 40 for edits."),
        io.Float.Input("cfg", default=1.0, min=0.0, max=20.0, step=0.1),
        io.Combo.Input("sampler_name", options=_samplers.KSampler.SAMPLERS, default="euler"),
        io.Combo.Input("scheduler", options=_samplers.KSampler.SCHEDULERS, default="simple"),
    ]


def _bundle(tag2pinfo, H, W):
    return {"tag2pinfo": tag2pinfo, "frame_size": (int(H), int(W))}


def _full(info, H, W):
    """PARTS_BUNDLE entry -> full-frame RGBA."""
    out = np.zeros((H, W, 4), np.uint8)
    x1, y1, x2, y2 = info["xyxy"]
    img = info["img"]
    if img.shape[:2] != (y2 - y1, x2 - x1):
        img = np.asarray(C.Image.fromarray(img).resize((x2 - x1, y2 - y1), C.Image.LANCZOS))
    out[y1:y2, x1:x2] = img
    return out


def _to_source(a, fh, fw, Hs, Ws):
    """A full-frame array from a bundle frame (fh, fw) into the source frame (Hs, Ws) - same rule as the plates:
    both frames are the same picture padded to square, scaled."""
    if (fh, fw) == (Hs, Ws):
        return a
    return C.to_frame(a, [0, 0, max(fh, fw)], max(Hs, Ws))[:Hs, :Ws]


# ═════════════════════════════════════════════════════════════════════════════
class BD_PartsVocabulary(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsVocabulary",
            display_name="BD Parts Vocabulary",
            category=CATEGORY,
            description=(
                "The CLOSED list of items BD Parts Builder 2 may extract, plus the matching true/false prompt for a "
                "vision model (wire vlm_prompt into AILab_QwenVL custom_prompt, its answer into BD Parts Builder 2 "
                "presence, and vocabulary into its vocabulary). Qwen Image 2.1 draws anything it is asked for, "
                "present or not, so only items the vote finds are ever requested.\n\n"
                "One line per item:  key | words the vision model is asked | noun used in edit prompts | "
                "accessory (yes/no). Accessories are what 'head without items' and the bald plate remove."),
            inputs=[io.String.Input("vocabulary", multiline=True, default=C.DEFAULT_VOCABULARY)],
            outputs=[io.String.Output(display_name="vocabulary"), io.String.Output(display_name="vlm_prompt")],
        )

    @classmethod
    def execute(cls, vocabulary) -> io.NodeOutput:
        voc = C.parse_vocabulary(vocabulary)
        return io.NodeOutput(vocabulary, C.vlm_prompt(voc))


# ═════════════════════════════════════════════════════════════════════════════
class BD_PartsBuilder2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsBuilder2",
            display_name="BD Parts Builder 2 (Qwen 2.1)",
            category=CATEGORY,
            description=(
                "Decompose a character head into asset layers with Qwen Image 2.1 prompts - no SAM, frame-true.\n\n"
                "For every item the presence vote found:\n"
                "  VISIBLE  - 'extract only the X exactly where it is' (RGBA). Trusted for WHERE, not for pixels: each "
                "source pixel goes to the faithful edit that matches it, and the layer is the SOURCE's own pixels, "
                "so the visible layers rebuild the source exactly. Items whose edit does not match the source "
                "(hidden, or restyled) own nothing and keep only a complete layer.\n"
                "  COMPLETE - 'decompose the head into layers, output only the X layer': the whole object, hidden "
                "parts included. Split into front and _back (pixels over bare skin are behind the head).\n"
                "Paint order comes from the source: on two complete layers' overlap, the one the source shows is in "
                "front (depth only breaks ties). depth_median in both bundles encodes that order for "
                "BD_PartsExport.\n\n"
                "base_image = head without items (the voted accessories removed), frame-proven with seed retries; "
                "feed it to BD_PartsExport base_image with parts to get a PSD that rebuilds the source."),
            inputs=[
                io.Image.Input("image", tooltip="Headshot (e.g. HeadMasters MASTER 1). Its RGB is used as given."),
                io.Model.Input("model", tooltip="Qwen Image 2.1 through QwenImage21Cache."),
                io.Clip.Input("clip", tooltip="qwen3vl_8b, CLIPLoader type qwen_image."),
                io.Vae.Input("vae", tooltip="Qwen Image 2.1 VAE (RGBA)."),
                io.String.Input("presence", multiline=True, default="",
                                tooltip="The vision model's JSON answer (AILab_QwenVL with BD Parts Vocabulary's "
                                        "vlm_prompt), or a typed list: 'hair, eyes, mouth, hat'."),
                io.String.Input("vocabulary", multiline=True, default=C.DEFAULT_VOCABULARY,
                                tooltip="From BD Parts Vocabulary. key | ask | noun | accessory."),
                io.Mask.Input("head_mask", optional=True,
                              tooltip="Where the head is (1 = head), e.g. the image's alpha. None: non-white pixels."),
                io.Image.Input("depth_image", optional=True,
                               tooltip="BD Lotus2 Predict map or raw_linear - BRIGHTER = NEARER. Breaks paint-order "
                                       "ties only."),
                *_sampler_inputs(),
                io.Boolean.Input("make_complete", default=True, tooltip="Also make the COMPLETE layers."),
                io.Boolean.Input("make_base", default=True, tooltip="Also make the head without items."),
                io.Boolean.Input("colour_retry", default=True,
                                 tooltip="One retry with the measured colour named ('the white sunglasses') for an "
                                         "ACCESSORY whose edit does not match the source. Never for hair/brows/eyes/"
                                         "mouth: next to an accessory their measured colour is the accessory's."),
                io.Int.Input("base_attempts", default=3, min=1, max=8,
                             tooltip="Head without items: seeds tried (seed, seed+1, ...) until it is frame-true."),
                io.Float.Input("fidelity_min", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Share of an edit's pixels that must match the source (< 40 levels) for it to "
                                       "own pixels."),
                io.Float.Input("explained_max", default=60.0, min=1.0, max=255.0, step=1.0,
                               tooltip="A pixel is explained by an edit when their colours are within this many levels."),
            ],
            outputs=[
                io.Custom(PARTS_BUNDLE).Output(display_name="parts"),
                io.Custom(PARTS_BUNDLE).Output(display_name="complete_parts"),
                io.Image.Output(display_name="base_image"),
                io.Image.Output(display_name="layers_preview"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, image, model, clip, vae, presence, vocabulary, head_mask=None, depth_image=None,
                seed=20260931, steps=40, cfg=1.0, sampler_name="euler", scheduler="simple", make_complete=True,
                make_base=True, colour_retry=True, base_attempts=3, fidelity_min=0.5,
                explained_max=60.0) -> io.NodeOutput:
        src, head0 = _source(image, head_mask)
        H, W = src.shape[:2]
        voc = C.parse_vocabulary(vocabulary)
        byk = {v["key"]: v for v in voc}
        vote = C.parse_vote(presence, voc)
        present = [v["key"] for v in voc if vote.get(v["key"])]
        near = None
        if depth_image is not None:
            d = _img_u8(depth_image)[..., :3].astype(np.float32).mean(-1) / 255.0
            near = d if d.shape == (H, W) else np.asarray(C.Image.fromarray(_u8(d)).resize((W, H))) / 255.0
        q = _Q21(model, clip, vae, steps, cfg, sampler_name, scheduler)
        n_edits = len(present) * (2 if make_complete else 1) + (1 if make_base else 0)
        pbar = comfy.utils.ProgressBar(max(n_edits, 1))

        edits, complete = {}, {}
        for k in present:
            edits[k] = q(src, C.visible_prompt(byk[k]["noun"]), seed)
            pbar.update(1)
            if make_complete:
                complete[k] = q(src, C.complete_prompt(byk[k]["noun"]), seed)
                pbar.update(1)
        R = C.resolve(src, head0, edits, fidelity_min, explained_max)
        retried = {}
        if colour_retry:
            for k in present:
                p = R["parts"][k]
                if byk[k]["accessory"] and k not in C.NEVER_RECOLOUR and p["hidden_or_restyled"] and p["edit_colour"]:
                    edits[k] = q(src, C.visible_prompt(byk[k]["noun"], p["edit_colour"]), seed)
                    retried[k] = {"visible": p["edit_colour"]}
            if retried:
                R = C.resolve(src, head0, edits, fidelity_min, explained_max)
            if make_complete:
                for k in present:
                    p = R["parts"][k]
                    shift = C.complete_colour_shift(complete[k], src, p["own"])
                    if byk[k]["accessory"] and (shift or 0) > 45 and p["colour_of_item"]:
                        complete[k] = q(src, C.complete_prompt(byk[k]["noun"], p["colour_of_item"]), seed)
                        retried.setdefault(k, {})["complete"] = p["colour_of_item"]
        head = R["head"]
        owner = R["owner"]
        skin0 = head & (owner == 0)

        # paint-order tie-break: Lotus2 is brighter = nearer, so far = 1 - median; an item that owns nothing is
        # hidden behind something and sorts farthest (the proven runner's -1 on its nearness scale)
        far, near_med = {}, {}
        for k in present:
            own = R["parts"][k]["own"]
            if not own.any():
                far[k] = 2.0
            elif near is not None:
                near_med[k] = float(np.median(near[own]))
                far[k] = 1.0 - near_med[k]
            else:
                far[k] = 0.5
        if make_complete:
            order, evidence = C.paint_order(present, complete, owner, far)
        else:
            order, evidence = sorted(present, key=lambda k: -far[k]), {}
        order, headwear_moved = C.headwear_over_hair(order)
        dvis = C.order_depths(order)

        tag2 = {}
        for i, k in enumerate(order):
            p = R["parts"][k]
            info = C.part_info(k, p["visible"], dvis[k], near, fidelity=p["fidelity"], paint_index=i,
                               hidden=bool(p["hidden_or_restyled"]), accessory=bool(byk[k]["accessory"]),
                               depth_near_median=near_med.get(k), edit_cover=p["edit_cover"].astype(np.uint8) * 255)
            if info is not None:
                tag2[k] = info
        parts = _bundle(tag2, H, W)

        ctag2, back_px, wins = {}, {}, {}
        n = max(len(order), 1)
        if make_complete:
            for i, k in enumerate(order):
                B, F, back_px[k] = C.split_back_front(complete[k], skin0)
                F, wins[k] = C.visible_wins(F, R["parts"][k]["own"], [R["parts"][s]["own"] for s in order[:i]], src)
                common = dict(item=k, paint_index=i, accessory=bool(byk[k]["accessory"]),
                              edit_cover=R["parts"][k]["edit_cover"].astype(np.uint8) * 255)
                fb = C.part_info(k + "_back", B, 1.0 - (i + 0.5) / (2 * n), near, role="back", **common)
                ff = C.part_info(k, F, 0.5 - (i + 0.5) / (2 * n), near, role="front", **common)
                for info in (fb, ff):
                    if info is not None:
                        ctag2[info["tag"]] = info
        complete_parts = _bundle(ctag2, H, W)

        # head without items - frame-proven over the head no item covers; seed, seed+1, ... until it holds
        base, base_frame, base_seed, base_prompt = src, {"identical": True, "pass": True}, None, None
        acc = [k for k in present if byk[k]["accessory"]]
        if make_base and acc:
            keep = [byk[k]["noun"] for k in present if not byk[k]["accessory"]]
            base_prompt = C.base_prompt([byk[k]["noun"] for k in acc], keep)
            cover_any = np.zeros((H, W), bool)
            for k in present:
                cover_any |= R["parts"][k]["edit_cover"]
            region = head & ~R["owned_any"] & ~cover_any
            tries = []
            for i in range(base_attempts):
                cand = q(src, base_prompt, seed + i)[..., :3]
                fr = C.register(src, cand, region)
                tries.append(dict(fr, seed=seed + i))
                base, base_frame, base_seed = cand, fr, seed + i
                if fr.get("pass") is not False:          # True, or None = too little uncovered head to measure
                    break
            base_frame = dict(base_frame, attempts=tries)
        pbar.update(1)

        prev = [p["visible"] for p in R["parts"].values()] + [complete[k] for k in order if k in complete]
        report = {"node": "BD_PartsBuilder2", "vote": vote, "present": present, "paint_order_back_to_front": order,
                  "paint_order_evidence": evidence, "headwear_moved_over_hair": headwear_moved,
                  "visible_wins": wins, "edits": q.count, "attention": "cuDNN backend excluded", "colour_retries": retried,
                  "unexplained_cover": round(R["unexplained"], 4),
                  "alpha_repaired_px": int((head & ~head0).sum()),
                  "base_frame": base_frame, "base_seed": base_seed,
                  "parts": {k: {"fidelity": R["parts"][k]["fidelity"], "hidden_or_restyled": R["parts"][k]["hidden_or_restyled"],
                                "visible_px": int(R["parts"][k]["own"].sum()), "complete_back_px": back_px.get(k),
                                "depth_near_median": near_med.get(k)} for k in present},
                  "prompts": {"visible": C.visible_prompt("X"), "complete": C.complete_prompt("X"), "base": base_prompt}}
        print("[BD PartsBuilder2] present=%s order=%s edits=%d base_frame=%s" % (present, order, q.count,
              {k: base_frame.get(k) for k in ("pass", "scale_x", "centre_move_px")}), flush=True)
        preview = _t(prev) if prev else _t(np.zeros((H, W, 4), np.uint8))
        return io.NodeOutput(parts, complete_parts, _t(base), preview, json.dumps(report, indent=1))


# ═════════════════════════════════════════════════════════════════════════════
class BD_PartsBuilder2Plates(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsBuilder2Plates",
            display_name="BD Parts Builder 2 Plates (Qwen 2.1)",
            category=CATEGORY,
            description=(
                "The featureless head plates, frame-true: bald (RGBA edit of the source; its alpha is the plate "
                "MATTE) -> mouthless/browless and eyeless, each ONE edit of the bald, every stage cut by the matte "
                "onto pure white before anything reads it.\n\n"
                "Every stage must register onto its input AND the source (coarse-to-fine affine ECC over the head "
                "minus what the stage removes): |scale-1| <= max_scale_err, centre move <= max_offset_px. A weak "
                "direct fit (ecc < 0.7, e.g. a laugh's cheeks relaxing) is proven through the chain. A failed stage "
                "retries with the next seed; if every attempt fails, the chain STOPS (later plates are white and "
                "plate_frame says ok=false, so Assemble builds nothing from a rejected plate).\n\n"
                "Plate frame: if the bald head fills less than reframe_below of the frame (a hat set the source's "
                "framing), the source is cropped to the bald head and the chain re-runs at full resolution. "
                "plate_frame (JSON) records the crop; plate_source / plate_head are the source in that frame. "
                "work_size 2048 runs the edits at 2048 (Qwen Image 2.1's max) instead."),
            inputs=[
                io.Image.Input("image", tooltip="The same source as BD Parts Builder 2."),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Custom(PARTS_BUNDLE).Input("parts", optional=True,
                                              tooltip="BD Parts Builder 2 parts - where each item is (frame proof)."),
                io.Custom(PARTS_BUNDLE).Input("complete_parts", optional=True,
                                              tooltip="BD Parts Builder 2 complete_parts (frame proof)."),
                io.Mask.Input("head_mask", optional=True),
                io.String.Input("bald_prompt", multiline=True, default=C.BALD_PROMPT),
                io.String.Input("mouthless_prompt", multiline=True, default=C.MOUTHLESS_PROMPT),
                io.String.Input("eyeless_prompt", multiline=True, default=C.EYELESS_PROMPT),
                *_sampler_inputs(20260933),
                io.Int.Input("attempts_per_stage", default=2, min=1, max=8,
                             tooltip="Seeds tried per stage (seed, seed+1, ...) before the chain stops."),
                io.Combo.Input("work_size", options=["1024", "2048"], default="1024",
                               tooltip="Plate resolution. 2048 is Qwen Image 2.1's maximum (about 4x the time)."),
                io.Float.Input("reframe_below", default=0.90, min=0.0, max=1.0, step=0.01,
                               tooltip="Re-frame when the bald head fills less than this of the frame. 0 = never."),
                io.Float.Input("reframe_fill", default=0.94, min=0.5, max=0.99, step=0.01),
                io.Float.Input("max_scale_err", default=0.015, min=0.001, max=0.2, step=0.001),
                io.Float.Input("max_offset_px", default=3.0, min=0.1, max=50.0, step=0.1,
                               tooltip="Centre move limit, in 1024-px units."),
                io.Combo.Input("lip_zone", options=["organic", "contour", "hull", "plane"], default="organic",
                               optional=True,
                               tooltip="Lip shape in socket_mask / feature_mask. organic: the MediaPipe outer lip "
                                       "contour + 6 px, like the eyes and brows. contour: the exact lip outline, no "
                                       "margin. hull: its convex hull (no cupid's bow). plane: the box FaceMaker "
                                       "v10-v26 drew for its lip stamp."),
                io.Combo.Input("eye_fill", options=["flat", "none"], default="flat", optional=True,
                               tooltip="flat: each MediaPipe eye zone of the eyeless plate becomes one flat tone (the "
                                       "median of the skin around it) - the flat eye section the engine eye sits on, "
                                       "which FaceMaker v26 made in its removed 'Crop and Fill Eyes' step. none: keep "
                                       "the closed lids 2.1 draws."),
            ],
            outputs=[
                io.Image.Output(display_name="bald"),
                io.Image.Output(display_name="mouthless"),
                io.Image.Output(display_name="eyeless"),
                io.Mask.Output(display_name="matte"),
                io.Image.Output(display_name="plate_source"),
                io.Mask.Output(display_name="plate_head"),
                io.String.Output(display_name="plate_frame"),
                io.String.Output(display_name="report"),
                io.Mask.Output(display_name="socket_mask",
                               tooltip="Eye / brow / mouth exclusion mask (MediaPipe on the bald plate; lips per "
                                       "lip_zone), in the plate frame. Save it with the plates; FaceMaker subtracts "
                                       "it from the matte instead of detecting."),
                io.Image.Output(display_name="feature_mask",
                                tooltip="RGB: R = mouth / lips, G = eyes, B = brows (plate frame)."),
            ],
        )

    @classmethod
    def execute(cls, image, model, clip, vae, parts=None, complete_parts=None, head_mask=None,
                bald_prompt=C.BALD_PROMPT, mouthless_prompt=C.MOUTHLESS_PROMPT, eyeless_prompt=C.EYELESS_PROMPT,
                seed=20260933, steps=40, cfg=1.0, sampler_name="euler", scheduler="simple", attempts_per_stage=2,
                work_size="1024", reframe_below=0.90, reframe_fill=0.94, max_scale_err=0.015,
                max_offset_px=3.0, lip_zone="organic", eye_fill="flat") -> io.NodeOutput:
        src, head = _source(image, head_mask)
        H, W = src.shape[:2]
        size = int(work_size)
        q = _Q21(model, clip, vae, steps, cfg, sampler_name, scheduler)
        pbar = comfy.utils.ProgressBar(4)

        # where each item is, in the source frame: its complete layer off bare skin + its visible edit's cover
        lay, lay_vis, accessory = {}, {}, set()
        for b in (parts, complete_parts):
            if b is None:
                continue
            visible_bundle = b is parts
            b = ensure_bundle(b, source="BD_PartsBuilder2Plates")
            fh, fw = b.get("frame_size") or (H, W)
            for tag, info in b["tag2pinfo"].items():
                key = info.get("item", tag)
                if info.get("accessory"):
                    accessory.add(key)
                m = np.zeros((fh, fw), bool)
                if info.get("role") != "back":
                    m |= _full(info, fh, fw)[..., 3] > 127
                ec = info.get("edit_cover")
                if ec is not None and ec.shape == (fh, fw):
                    m |= ec > 127
                ms = _to_source(m, fh, fw, H, W)
                lay[key] = lay.get(key, np.zeros((H, W), bool)) | ms
                if visible_bundle:
                    lay_vis[key] = lay_vis.get(key, np.zeros((H, W), bool)) | ms

        def removed(stage, layers=None):
            layers = lay if layers is None else layers
            keys = set()
            for s in ("bald", stage):
                for t in C.REMOVES[s]:
                    keys |= accessory if t == "__accessories__" else {t}
            g = np.zeros((H, W), bool)
            for t in keys:
                if t in layers:
                    g |= layers[t]
            return g

        prompts = {"bald": bald_prompt, "mouthless": mouthless_prompt, "eyeless": eyeless_prompt}
        made_from = {"bald": "source", "mouthless": "bald", "eyeless": "bald"}
        report = {"node": "BD_PartsBuilder2Plates", "stages": {}, "plate_frame": None, "ok": True, "failed_stage": None}
        k = size / 1024.0
        dil = int(round(31 * k)) | 1

        def chain(fsrc, fhead, box, stages, done, matte):
            for stage in stages:
                prev = done[made_from[stage]]
                g = C.to_frame(removed(stage), box, size)
                region = fhead & ~C.cv2.dilate(g.astype(np.uint8), np.ones((dil, dil), np.uint8)).astype(bool)
                gv = C.to_frame(removed(stage, lay_vis), box, size) if lay_vis else None
                region_vis = None
                if gv is not None:
                    region_vis = fhead & ~C.cv2.dilate(gv.astype(np.uint8), np.ones((dil, dil), np.uint8)).astype(bool)
                ok, got = False, None
                for i in range(attempts_per_stage):
                    raw = q(prev, prompts[stage], seed + i)
                    if stage == "bald":
                        matte = C.clamp_alpha(raw[..., 3].astype(np.float32) / 255.0)
                    got = C.over_white(raw, matte)
                    fp = C.register(prev, got, region, max_scale_err, max_offset_px)
                    fm = C.register(fsrc, got, region, max_scale_err, max_offset_px)
                    ok, how = bool(fp.get("pass")) and bool(fm.get("pass")), "direct"
                    if not ok and region_vis is not None and C.unmeasured(fp, fm):
                        # the complete layers' hidden extents left too little head to register on (a cap, dark
                        # glasses and a full beard cover nearly the whole face): prove on the head minus what is
                        # VISIBLY removed. Only when ECC could not measure - a measured miss is never re-tried.
                        fp = dict(C.register(prev, got, region_vis, max_scale_err, max_offset_px), region="visible items")
                        fm = dict(C.register(fsrc, got, region_vis, max_scale_err, max_offset_px), region="visible items")
                        ok, how = bool(fp.get("pass")) and bool(fm.get("pass")), "direct, visible-item region"
                    up = report["stages"].get(made_from[stage], {})
                    if not ok and fp.get("pass") and made_from[stage] != "source" and (fm.get("ecc") or 0) < 0.7:
                        ch = C.chain_proof(up.get("vs_source", {}), fp, max_scale_err, max_offset_px)
                        fm = dict(fm, chain=ch)
                        ok, how = ch["pass"], "chain (direct fit weak: ecc %.2f)" % (fm.get("ecc") or 0)
                    report["stages"][stage] = {"seed": seed + i, "vs_input": fp, "vs_source": fm, "pass": ok,
                                               "pass_by": how if ok else None}
                    if ok:
                        break
                pbar.update(1)
                if not ok:                               # stop: nothing is made from a rejected plate
                    report["ok"], report["failed_stage"] = False, stage
                    done[stage + "_rejected"] = got
                    print("[BD PartsBuilder2Plates] %s failed its frame proof - chain stopped: %s"
                          % (stage, report["stages"][stage]), flush=True)
                    return done, matte
                done[stage] = got
            return done, matte

        box0 = None if (H == W == size) else [0, 0, max(H, W)]
        fsrc0, fhead0 = C.to_frame(src, box0, size), C.to_frame(head, box0, size)
        done, matte = chain(fsrc0, fhead0, box0, ["bald"], {"source": fsrc0}, None)
        box, fsrc, fhead = box0, fsrc0, fhead0
        if report["ok"]:
            # plate frame: the bald's matte, mapped back to source px, sets the crop
            m_src = matte if box0 is None else \
                C.to_frame(_u8(matte), [0, 0, size], max(H, W))[:H, :W].astype(np.float32) / 255.0
            nb = C.plate_box(m_src, reframe_below, reframe_fill) if reframe_below > 0 else None
            if nb:
                report["stages"]["source_frame_bald"] = report["stages"].pop("bald")
                box, fsrc, fhead = nb, C.to_frame(src, nb, size), C.to_frame(head, nb, size)
                done, matte = chain(fsrc, fhead, box, ["bald", "mouthless", "eyeless"], {"source": fsrc}, None)
            else:
                done, matte = chain(fsrc, fhead, box, ["mouthless", "eyeless"], done, matte)
        white = np.full((size, size, 3), 255, np.uint8)
        if matte is None:
            matte = np.zeros((size, size), np.float32)
        frame = {"box_source_px": box, "size": size, "source_size": [W, H],
                 "scale": round(size / box[2], 5) if box else 1.0,
                 "rule": "plate_px = (source_px - box_xy) * size / box_side" if box else "identity",
                 "ok": report["ok"], "failed_stage": report["failed_stage"]}
        report["plate_frame"] = frame
        report["prompts"] = prompts
        report["attention"] = "cuDNN backend excluded"
        out = [done.get(s, done.get(s + "_rejected", white)) for s in ("bald", "mouthless", "eyeless")]
        socket, feat = np.zeros((size, size), np.float32), np.zeros((size, size, 3), np.uint8)
        if "bald" in done:                                   # a frame-proven bald: bald, not faceless
            try:
                socket, feat, st = _feature_masks(done["bald"], lip_zone)
                report["feature_mask"] = {"status": st[:200], "socket_px": int((socket > 0.5).sum()),
                                          "channels": "R = mouth/lips, G = eyes, B = brows",
                                          "lip_zone": lip_zone,
                                          "settings": "BD_FaceSocketInfill as FaceMaker v10-v26 #8580, lips %s" % lip_zone}
            except Exception as e:                           # no face found etc. - the plates still stand
                report["feature_mask"] = {"error": str(e)[:200]}
        if eye_fill == "flat" and "eyeless" in done and (feat[..., 1] > 127).any():
            out[2], tones = C.flat_fill(out[2], feat[..., 1] > 127, np.asarray(matte) > 0.5)
            report["eye_fill"] = {"mode": "flat", "zones": tones}
        print("[BD PartsBuilder2Plates] ok=%s frame=%s edits=%d feature_mask=%s" % (report["ok"], frame, q.count,
              report.get("feature_mask", {}).get("socket_px", report.get("feature_mask"))), flush=True)
        return io.NodeOutput(_t(out[0]), _t(out[1]), _t(out[2]), _m(matte), _t(fsrc),
                             _m(fhead.astype(np.float32)), json.dumps(frame), json.dumps(report, indent=1),
                             _m(socket), _t(feat))


# ═════════════════════════════════════════════════════════════════════════════
class BD_PartsBuilder2Assemble(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsBuilder2Assemble",
            display_name="BD Parts Builder 2 Assemble",
            category=CATEGORY,
            description=(
                "The engine stack: every complete layer's BACK part (behind the head) -> the featureless eyeless "
                "plate over its matte -> every FRONT part, in the paint order BD Parts Builder 2 read off the source "
                "- and how far that lands from the source. engine_parts is a PARTS_BUNDLE in the plate frame "
                "(back parts, 'head_plate', front parts; depth_median = stack order) for BD_PartsExport, which "
                "writes it as a PSD. Builds nothing when the plates failed their frame proof."),
            inputs=[
                io.Custom(PARTS_BUNDLE).Input("complete_parts"),
                io.Image.Input("plate", tooltip="BD Parts Builder 2 Plates eyeless."),
                io.Mask.Input("matte", tooltip="BD Parts Builder 2 Plates matte."),
                io.String.Input("plate_frame", force_input=True, tooltip="BD Parts Builder 2 Plates plate_frame."),
                io.Image.Input("plate_source", tooltip="BD Parts Builder 2 Plates plate_source."),
                io.Mask.Input("plate_head", optional=True, tooltip="BD Parts Builder 2 Plates plate_head."),
            ],
            outputs=[
                io.Custom(PARTS_BUNDLE).Output(display_name="engine_parts"),
                io.Image.Output(display_name="reassembled"),
                io.Image.Output(display_name="difference"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, complete_parts, plate, matte, plate_frame, plate_source, plate_head=None) -> io.NodeOutput:
        cp = ensure_bundle(complete_parts, source="BD_PartsBuilder2Assemble")
        fh, fw = cp.get("frame_size")
        pl = _img_u8(plate)[..., :3]
        size = pl.shape[0]
        psrc = _img_u8(plate_source)[..., :3]
        if not plate_frame:
            raise ValueError("BD Parts Builder 2 Assemble: plate_frame is empty - wire BD Parts Builder 2 Plates "
                             "plate_frame, or the layers cannot be placed on the plate.")
        frame = json.loads(plate_frame)
        if frame.get("size") not in (None, size):
            raise ValueError("BD Parts Builder 2 Assemble: plate is %d px but plate_frame says %s"
                             % (size, frame.get("size")))
        if frame.get("ok") is False:
            rep = {"node": "BD_PartsBuilder2Assemble", "ok": False,
                   "why": "the plates failed their frame proof at '%s' - nothing is built from a rejected plate "
                          "(see the Plates report; try another seed)" % frame.get("failed_stage")}
            print("[BD PartsBuilder2Assemble] %s" % rep["why"], flush=True)
            blank = np.full((size, size, 3), 255, np.uint8)
            return io.NodeOutput(_bundle({}, size, size), _t(blank), _t(blank), json.dumps(rep, indent=1))
        box = frame.get("box_source_px")
        Ws, Hs = frame.get("source_size") or [fw, fh]
        mt = (matte[0] if matte.ndim == 3 else matte).detach().cpu().float().numpy()
        phead = _mask_bool(plate_head, psrc.shape[:2]) if plate_head is not None else (psrc.astype(np.int16).min(-1) < 245)

        items = {}
        for tag, info in cp["tag2pinfo"].items():
            item = info.get("item", tag[:-5] if tag.endswith("_back") else tag)
            full = _to_source(_full(info, fh, fw), fh, fw, Hs, Ws)
            full = C.to_frame(full, box, size)          # box None: pad to square + scale, as the plates did
            slot = items.setdefault(item, {"paint_index": info.get("paint_index", 0),
                                           "layers": [np.zeros((size, size, 4), np.uint8)] * 2})
            if info.get("role") == "back":
                slot["layers"] = [full, slot["layers"][1]]
            else:
                slot["layers"] = [slot["layers"][0], full]
            if "paint_index" in info:
                slot["paint_index"] = info["paint_index"]
        order = sorted(items, key=lambda t: items[t]["paint_index"])           # back -> front
        comp, heat, rep = C.reassemble(pl, mt, {t: items[t]["layers"] for t in order}, order, psrc, phead)

        n = max(len(order), 1)
        tag2 = {}
        for i, t in enumerate(order):
            b = C.part_info(t + "_back", items[t]["layers"][0], 1.0 - (i + 0.5) / (2 * n) * 0.98, role="back",
                            item=t, paint_index=i)
            f = C.part_info(t, items[t]["layers"][1], 0.49 - (i + 0.5) / (2 * n) * 0.98, role="front",
                            item=t, paint_index=i)
            for info in (b, f):
                if info is not None:
                    tag2[info["tag"]] = info
        hp = C.part_info("head_plate", np.dstack([pl, _u8(np.clip(mt, 0, 1))]), 0.495, role="plate")
        if hp is not None:
            tag2["head_plate"] = hp
        rep.update(node="BD_PartsBuilder2Assemble", ok=True, paint_order_back_to_front=order, plate_frame=frame)
        print("[BD PartsBuilder2Assemble] diff_on_head=%s order=%s" % (rep["diff_on_head"], order), flush=True)
        return io.NodeOutput(_bundle(tag2, size, size), _t(comp), _t(heat), json.dumps(rep, indent=1))


# ═════════════════════════════════════════════════════════════════════════════
class BD_CompareImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CompareImages",
            display_name="BD Compare Images",
            category=CATEGORY,
            description=(
                "How far a candidate lands from a reference - the red-zone difference the head chain is judged by. "
                "difference: red = per-pixel |reference - candidate| x gain, the region tinted teal. side_by_side: "
                "reference | candidate | difference with the score under it. score: mean difference on the region "
                "(0 = identical, in 0-255 levels) and the share of the region more than `threshold` levels off.\n\n"
                "Use it wherever a result should reproduce a source: the visible-layer PSD composite against the "
                "headshot, the engine stack against the plate-frame source, a plate against the image it was made "
                "from. A candidate with alpha is laid over white first; a different size is resized to the reference."),
            inputs=[
                io.Image.Input("reference", tooltip="What the candidate should reproduce."),
                io.Image.Input("candidate", tooltip="The result being judged."),
                io.Mask.Input("region", optional=True,
                              tooltip="Where to measure (1 = measure). None: pixels that are not white in either image."),
                io.Float.Input("gain", default=3.0, min=0.5, max=20.0, step=0.5,
                               tooltip="Red = difference x gain (3: 85 levels off is full red)."),
                io.Int.Input("threshold", default=40, min=1, max=255, tooltip="Levels off that count as 'off' in the score."),
                io.String.Input("label", default="", optional=True, tooltip="Shown on the side_by_side panel."),
            ],
            outputs=[
                io.Image.Output(display_name="difference"),
                io.Image.Output(display_name="side_by_side"),
                io.String.Output(display_name="score"),
            ],
        )

    @classmethod
    def execute(cls, reference, candidate, region=None, gain=3.0, threshold=40, label="") -> io.NodeOutput:
        ref = _img_u8(reference)
        cand = _img_u8(candidate)
        ref = C.over_white(ref) if ref.shape[-1] == 4 else ref[..., :3]
        cand = C.over_white(cand) if cand.shape[-1] == 4 else cand[..., :3]
        H, W = ref.shape[:2]
        if cand.shape[:2] != (H, W):
            cand = np.asarray(C.Image.fromarray(cand).resize((W, H), C.Image.LANCZOS))
        reg = _mask_bool(region, (H, W))
        if reg is None:
            reg = (ref.astype(np.int16).min(-1) < 245) | (cand.astype(np.int16).min(-1) < 245)
        heat, score, _ = C.compare(ref, cand, reg, gain, threshold)
        score = dict(score, label=label or None, threshold=threshold, gain=gain)
        foot = "%s  mean diff %s  |  %s%% of the region > %d levels off" % (
            (label + ":") if label else "", score["diff_on_region"],
            None if score["share_over_%d" % threshold] is None else round(100 * score["share_over_%d" % threshold], 1),
            threshold)
        panel = C.side_by_side(ref, cand, heat, footer=foot.strip())
        print("[BD CompareImages] %s" % foot.strip(), flush=True)
        return io.NodeOutput(_t(heat), _t(panel), json.dumps(score, indent=1))


class BD_EvenLight(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_EvenLight",
            display_name="BD Even Light",
            category=CATEGORY,
            description=(
                "Even a head's lighting without flattening its facets. The light is the head's luminance blurred "
                "heavily (mask-normalised, sigma in px at 1024); the facets are what is left. Output = mean + "
                "keep_light x (light - mean) + facet_gain x facets, colour carried by ratio; outside the mask the "
                "image is unchanged.\n\nWhy: Qwen Image 2.1 reads a cel-shading prompt as one hard key light - a "
                "bright side and a shadow side - and prompts that ask for even light flatten the facets too. Put this "
                "after a 2.1 shading pass (FaceMaker's Qwen Cell Shaded) to keep its facet contrast and take most of "
                "the light out."),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Head mask (1 = head). None: pixels that differ from the corner colour."),
                io.Float.Input("keep_light", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Share of the lighting gradient kept. 0 = perfectly even, 1 = unchanged."),
                io.Float.Input("facet_gain", default=0.9, min=0.0, max=3.0, step=0.05,
                               tooltip="Facet contrast multiplier. 1 = unchanged."),
                io.Float.Input("sigma", default=60.0, min=5.0, max=400.0, step=5.0,
                               tooltip="Light blur in px at 1024 (scaled with the image). Larger than a facet."),
            ],
            outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, image, mask=None, keep_light=0.5, facet_gain=0.9, sigma=60.0) -> io.NodeOutput:
        rgb = _img_u8(image)[..., :3]
        H, W = rgb.shape[:2]
        head = _mask_bool(mask, (H, W))
        if head is None:
            corners = np.concatenate([rgb[:16, :16].reshape(-1, 3), rgb[:16, -16:].reshape(-1, 3)])
            head = np.abs(rgb.astype(np.int16) - np.median(corners, 0)).max(-1) > 12
        out, stats = C.even_light(rgb, head, keep_light, facet_gain, sigma)
        print("[BD EvenLight] %s" % stats, flush=True)
        return io.NodeOutput(_t(out), json.dumps(stats))


PARTS_BUILDER2_V3_NODES = [BD_PartsVocabulary, BD_PartsBuilder2, BD_PartsBuilder2Plates, BD_PartsBuilder2Assemble,
                           BD_CompareImages, BD_EvenLight]
PARTS_BUILDER2_NODES = {c.__name__: c for c in PARTS_BUILDER2_V3_NODES}
PARTS_BUILDER2_DISPLAY_NAMES = {
    "BD_PartsVocabulary": "BD Parts Vocabulary",
    "BD_PartsBuilder2": "BD Parts Builder 2 (Qwen 2.1)",
    "BD_PartsBuilder2Plates": "BD Parts Builder 2 Plates (Qwen 2.1)",
    "BD_PartsBuilder2Assemble": "BD Parts Builder 2 Assemble",
    "BD_CompareImages": "BD Compare Images",
    "BD_EvenLight": "BD Even Light",
}
