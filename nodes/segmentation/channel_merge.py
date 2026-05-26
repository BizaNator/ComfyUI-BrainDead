"""
BD_ChannelMerge — blend/composite a source image into specific channels of a base image.

Lets you modify individual RGBA channels of an existing texture pack using a
Photoshop-style blend mode, with the source's own alpha channel optionally acting as
a soft compositing mask (so background transparency is ignored automatically).

Primary use cases:
- Add a shadow map into the R channel of an existing pack without touching G/B/A
- Composite a character's skin texture (with alpha) onto a single channel,
  ignoring the transparent background
- Chain multiple BD_ChannelMerge nodes to progressively build up a texture pack
  one channel at a time

Blend modes mirror standard compositing math:
  replace   = overwrite target with source (standard paste)
  add       = clamp(base + src, 0, 1)   — brightens
  multiply  = base × src                — darkens proportionally; white = no change
  screen    = 1 - (1-base)(1-src)       — brightens; black = no change
  overlay   = multiply/screen split at 0.5
  darken    = min(base, src)
  lighten   = max(base, src)
  subtract  = clamp(base - src, 0, 1)   — darkens
  difference = |base - src|
  soft_light = gentle contrast (Pegtop formula)
"""

import torch

from comfy_api.latest import io

from ...utils.luma import LUMA_STANDARDS, LUMA_TOOLTIP, get_luma_weights


BLEND_MODES = [
    "replace",
    "add",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "subtract",
    "difference",
    "soft_light",
]

TARGET_CHANNEL_OPTIONS = ["R", "G", "B", "A", "RG", "RB", "GB", "RGB", "RGBA"]

# Maps target label → list of 0-based channel indices in the output tensor
_TARGET_IDX = {
    "R":    [0],
    "G":    [1],
    "B":    [2],
    "A":    [3],
    "RG":   [0, 1],
    "RB":   [0, 2],
    "GB":   [1, 2],
    "RGB":  [0, 1, 2],
    "RGBA": [0, 1, 2, 3],
}

SOURCE_FROM_OPTIONS = [
    "luma_bt709",   # weighted perceptual luma (recommended for greyscale sources)
    "channel_R",    # red channel of source
    "channel_G",
    "channel_B",
    "channel_A",    # alpha channel of source (handy for special effects)
    "average",      # simple (R+G+B)/3
    "max_rgb",      # max(R,G,B) — closest to "brightness"
]


# ─── internal helpers ────────────────────────────────────────────────────────

def _to_bhwc(t: torch.Tensor) -> torch.Tensor:
    return t if t.ndim == 4 else t.unsqueeze(0)


def _resize_to_hw(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize (B,H,W,C) or (B,H,W) to (B, h, w, C/∅) using bilinear."""
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    if t.ndim == 4:
        nchw = t.permute(0, 3, 1, 2)
        out = torch.nn.functional.interpolate(nchw, size=(h, w), mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1)
    if t.ndim == 3:
        out = torch.nn.functional.interpolate(t.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
        return out.squeeze(1)
    return t


def _mask_to_bhw1(mask: torch.Tensor, b: int, h: int, w: int,
                  dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Normalise any incoming MASK into (B, H, W, 1)."""
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, -1)
    elif m.ndim == 3:
        if m.shape[0] == 1 and b > 1:
            m = m.expand(b, -1, -1)
        m = m.unsqueeze(-1)
    elif m.ndim == 4:
        if m.shape[-1] > 1:
            m = m[..., 0:1]
    m = _resize_to_hw(m, h, w)
    return m.to(dtype=dtype, device=device)


def _extract_source_scalar(src: torch.Tensor, source_from: str,
                           luma_standard: str) -> torch.Tensor:
    """Collapse a (B,H,W,C) source to a (B,H,W,1) scalar using the chosen method."""
    nchan = src.shape[-1]

    if source_from == "channel_R":
        return src[..., 0:1]
    if source_from == "channel_G":
        return src[..., 1:2] if nchan >= 2 else src[..., 0:1]
    if source_from == "channel_B":
        return src[..., 2:3] if nchan >= 3 else src[..., 0:1]
    if source_from == "channel_A":
        if nchan >= 4:
            return src[..., 3:4]
        return torch.ones(src.shape[0], src.shape[1], src.shape[2], 1,
                          dtype=src.dtype, device=src.device)
    if source_from == "average":
        return src[..., :3].mean(dim=-1, keepdim=True) if nchan >= 3 else src[..., 0:1]
    if source_from == "max_rgb":
        return src[..., :3].max(dim=-1, keepdim=True).values if nchan >= 3 else src[..., 0:1]
    # default: luma_bt709
    weights = get_luma_weights(luma_standard).to(src.device).to(src.dtype)
    if nchan >= 3:
        return (src[..., :3] * weights).sum(dim=-1, keepdim=True)
    return src[..., 0:1]


def _apply_blend(base: torch.Tensor, src: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "add":
        return (base + src).clamp(0.0, 1.0)
    if mode == "multiply":
        return base * src
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - src)
    if mode == "overlay":
        return torch.where(base < 0.5,
                           2.0 * base * src,
                           1.0 - 2.0 * (1.0 - base) * (1.0 - src))
    if mode == "darken":
        return torch.minimum(base, src)
    if mode == "lighten":
        return torch.maximum(base, src)
    if mode == "subtract":
        return (base - src).clamp(0.0, 1.0)
    if mode == "difference":
        return (base - src).abs()
    if mode == "soft_light":
        # Pegtop formula: gentle S-curve contrast lift
        return (1.0 - 2.0 * src) * base * base + 2.0 * src * base
    return src  # replace


# ─── node ────────────────────────────────────────────────────────────────────

class BD_ChannelMerge(io.ComfyNode):
    """Blend a source image into specific channels of a base image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ChannelMerge",
            display_name="BD Channel Merge",
            category="🧠BrainDead/Segmentation",
            description=(
                "Blend/composite a source image into one or more specific RGBA channels "
                "of a base image, using Photoshop-style blend modes.\n\n"
                "Key feature: enable use_source_alpha to use the source image's own alpha "
                "as a soft mask — transparent areas of the source leave the base channel "
                "unchanged, so you can paste a character (with alpha background) onto a "
                "single channel without the background affecting it.\n\n"
                "Chain multiple BD_ChannelMerge nodes to build up a texture pack "
                "progressively, modifying one channel at a time as the image flows "
                "through the workflow."
            ),
            inputs=[
                io.Image.Input("base",
                               tooltip="Existing image to modify. Can be RGB or RGBA — the channels "
                                       "you don't target pass through unchanged."),
                io.Image.Input("source",
                               tooltip="Image to merge from. Resized to match base dimensions automatically."),
                io.Combo.Input("target_channels", options=TARGET_CHANNEL_OPTIONS, default="R",
                               tooltip="Which channel(s) of base to write to. "
                                       "Single: R, G, B, A — source is collapsed to a scalar (see source_from). "
                                       "Multi: RG, RB, GB, RGB, RGBA — source RGB channels map 1:1 to targets."),
                io.Combo.Input("source_from", options=SOURCE_FROM_OPTIONS, default="luma_bt709",
                               tooltip="For single-channel targets: how to collapse source to a scalar. "
                                       "luma_bt709 = perceptual greyscale (matches skin shader). "
                                       "channel_R/G/B/A = pick one channel directly. "
                                       "average = (R+G+B)/3. max_rgb = brightest channel. "
                                       "Ignored for multi-channel targets (RGB/RGBA) — source channels map directly."),
                io.Combo.Input("blend_mode", options=BLEND_MODES, default="replace",
                               tooltip="How blended source value combines with existing base channel value. "
                                       "replace = overwrite. add = brighten. multiply = darken proportionally. "
                                       "screen = brighten (black=no change). overlay = contrast split at 0.5. "
                                       "darken/lighten = min/max. subtract = darken clipped. "
                                       "difference = absolute gap. soft_light = gentle contrast."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Overall blend opacity. 0 = base unchanged; 1 = full blend applied. "
                                       "Combined multiplicatively with use_source_alpha mask and external mask."),
                io.Boolean.Input("use_source_alpha", default=True,
                                 tooltip="Use source image's alpha channel as a soft blend mask. "
                                         "Transparent source pixels (alpha=0) leave base unchanged; "
                                         "opaque pixels (alpha=1) apply the blend at full strength. "
                                         "This lets you composite a character against a transparent "
                                         "background without the BG affecting the target channel. "
                                         "No effect if source has no alpha (RGB only source)."),
                io.Boolean.Input("invert_alpha_mask", default=False, optional=True,
                                 tooltip="Invert the source alpha mask before using it. "
                                         "Blend applies where source IS transparent instead of opaque."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Additional external blend mask. Combined with source alpha and strength. "
                                      "White = full blend, black = base unchanged."),
                io.Combo.Input("luma_standard", options=LUMA_STANDARDS, default="bt709", optional=True,
                               tooltip="Luma weighting used when source_from=luma_bt709 (or any luma path). " + LUMA_TOOLTIP),
            ],
            outputs=[
                io.Image.Output(display_name="image",
                                tooltip="Modified base image. Same resolution as base. "
                                        "Will be RGBA if base was RGBA or if target_channels includes A; "
                                        "otherwise matches base channel count."),
            ],
        )

    @classmethod
    def execute(cls, base, source, target_channels="R", source_from="luma_bt709",
                blend_mode="replace", strength=1.0, use_source_alpha=True,
                invert_alpha_mask=False, mask=None, luma_standard="bt709") -> io.NodeOutput:

        base   = _to_bhwc(base).float()
        source = _to_bhwc(source).float()

        B, H, W, C_base = base.shape
        device, dtype = base.device, base.dtype

        # Resize source to match base spatial dimensions
        source = _resize_to_hw(source, H, W)
        # Batch-broadcast if needed
        if source.shape[0] == 1 and B > 1:
            source = source.expand(B, -1, -1, -1)

        target_idxs = _TARGET_IDX[target_channels]
        targets_A   = 3 in target_idxs

        # Expand base to RGBA if we need to write the alpha channel
        result = base.clone()
        if targets_A and C_base < 4:
            alpha_fill = torch.ones(B, H, W, 1, dtype=dtype, device=device)
            result = torch.cat([result, alpha_fill], dim=-1)

        C_out = result.shape[-1]

        # ── Build blend mask (B, H, W, 1) ──────────────────────────────────
        blend_mask = torch.full((B, H, W, 1), strength, dtype=dtype, device=device)

        if use_source_alpha and source.shape[-1] >= 4:
            src_alpha = source[..., 3:4]
            if invert_alpha_mask:
                src_alpha = 1.0 - src_alpha
            blend_mask = blend_mask * src_alpha

        if mask is not None:
            ext_mask = _mask_to_bhw1(mask, B, H, W, dtype, device)
            blend_mask = blend_mask * ext_mask

        blend_mask = blend_mask.clamp(0.0, 1.0)

        # ── Extract source values for target channels ───────────────────────
        n = len(target_idxs)
        if n == 1:
            src_vals = _extract_source_scalar(source, source_from, luma_standard)  # (B,H,W,1)
        else:
            # Multi-channel: map source RGB (and A if needed) directly to targets
            src_parts = []
            for idx in target_idxs:
                if idx < source.shape[-1]:
                    src_parts.append(source[..., idx:idx+1])
                else:
                    src_parts.append(torch.zeros(B, H, W, 1, dtype=dtype, device=device))
            src_vals = torch.cat(src_parts, dim=-1)  # (B,H,W,n)

        # ── Apply blend mode + composite with mask ──────────────────────────
        base_vals   = result[..., target_idxs]          # (B,H,W,n)
        blended     = _apply_blend(base_vals, src_vals, blend_mode)
        composited  = base_vals * (1.0 - blend_mask) + blended * blend_mask

        result[..., target_idxs] = composited.clamp(0.0, 1.0)

        # Drop the alpha channel if base didn't have one and we didn't target A
        if C_base == 3 and not targets_A:
            result = result[..., :3]

        print(
            f"[BD_ChannelMerge] target={target_channels} mode={blend_mode} "
            f"strength={strength:.2f} use_src_alpha={'yes' if use_source_alpha else 'no'} "
            f"base_C={C_base}→{result.shape[-1]} "
            f"mask_mean={float(blend_mask.mean()):.3f}"
        )

        return io.NodeOutput(result)


CHANNEL_MERGE_V3_NODES    = [BD_ChannelMerge]
CHANNEL_MERGE_NODES        = {"BD_ChannelMerge": BD_ChannelMerge}
CHANNEL_MERGE_DISPLAY_NAMES = {"BD_ChannelMerge": "BD Channel Merge"}
