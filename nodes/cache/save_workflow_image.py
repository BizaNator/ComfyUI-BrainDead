"""BD_SaveWorkflowImage — the workflow card, rendered and saved automatically.

ComfyUI can already export a picture of the canvas with the graph embedded, but
only by hand: right-click -> Export (PNG/SVG). There is no node, so it cannot be
part of a run. This is that node.

It does three things at once:

  1. Renders the BrainDead card — the SAME renderer used by
     tools/make_thumbnail.py (utils/thumbnail.py), so the template thumbnails and
     anything this node emits cannot drift apart. Every knob the CLI config
     accepts is exposed here as an input, including the watermark and its size,
     margin and corner.

  2. Takes its background either from an auto-drawn graph of the CURRENT workflow,
     or from an `image` input — a screenshot, a render, a photo of a tree —
     whichever you hand it, with the same card chrome over the top.

  3. Embeds the workflow + prompt into the PNG's text chunks the way ComfyUI's own
     SaveImage does, so the result opens by drag-and-drop.

On (3): the graph only reaches a node if the CLIENT sends it. The ComfyUI frontend
always does. A headless caller must post it explicitly:

    POST /prompt  {"prompt": {...}, "extra_data": {"extra_pnginfo": {"workflow": <UI graph>}}}

Without that, extra_pnginfo is empty, `workflow` is null, and the saved image will
NOT open in ComfyUI — so this node says so in its status string rather than
writing a file that looks fine and is not.

JPEG cannot carry those chunks. Choosing jpg gets the card but no embedded graph,
and the status says so. PNG is the default for that reason; the template
thumbnails that need `.jpg` come from the CLI, which never needed embedding.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy_api.latest import io

from ...utils import thumbnail
from ...utils.shared import OUTPUT_DIR


def _to_pil(image) -> Image.Image | None:
    """First frame of a ComfyUI IMAGE batch (B,H,W,C float 0..1) as RGB PIL."""
    if image is None:
        return None
    t = image[0] if isinstance(image, torch.Tensor) and image.dim() == 4 else image
    arr = np.clip(t.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr, "RGB")


def _lines(s):
    return [ln.strip() for ln in (s or "").splitlines() if ln.strip()]


def _csv(s):
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def _rgb(s, fallback):
    """'#aa78ff' or 'r,g,b' -> tuple. Falls back rather than raising on junk."""
    s = (s or "").strip()
    if not s:
        return fallback
    try:
        if s.startswith("#"):
            s = s.lstrip("#")
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
        parts = [int(p) for p in s.split(",")]
        if len(parts) == 3:
            return tuple(parts)
    except Exception:
        pass
    return fallback


def _logo_source(logo_image, logo_mask, mask_inverted, logo_path, notes):
    """RGBA PIL logo from the connected tensors, else the path, else the repo default.

    Returns whatever utils.thumbnail.watermark() accepts: a PIL Image or a path.
    """
    if logo_image is None:
        return logo_path.strip() or None
    rgb = _to_pil(logo_image)
    if rgb is None:
        return logo_path.strip() or None
    logo = rgb.convert("RGBA")
    if logo_mask is not None:
        try:
            m = logo_mask[0] if hasattr(logo_mask, "dim") and logo_mask.dim() == 3 else logo_mask
            a = np.clip(m.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if mask_inverted:
                a = 255 - a          # ComfyUI MASK is 1 where TRANSPARENT
            alpha = Image.fromarray(a, "L")
            if alpha.size != logo.size:
                alpha = alpha.resize(logo.size, Image.LANCZOS)
            logo.putalpha(alpha)
        except Exception as exc:
            notes.append("logo_mask ignored (%r)" % (exc,))
    else:
        notes.append("logo_image without logo_mask -- opaque rectangle")
    return logo


class BD_SaveWorkflowImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BD_SaveWorkflowImage",
            display_name="BD Save Workflow Image",
            category="BrainDead/save",
            description=(
                "Render the BrainDead card with the workflow graph (or your own image) "
                "as the background, stamp the title/version, and embed the workflow so "
                "the PNG opens in ComfyUI by drag-and-drop."
            ),
            inputs=[
                io.String.Input(
                    "filename_prefix", default="workflow_card", optional=True,
                    tooltip="Output name under the ComfyUI output dir, or an absolute path. "
                            "The extension is set by `format`.",
                ),
                io.String.Input(
                    "title", default="", multiline=False, optional=True,
                    tooltip="Big headline. Empty = the workflow's own name if one is known.",
                ),
                io.String.Input("subtitle", default="", multiline=False, optional=True),
                io.String.Input(
                    "wordmark", default="BrainDead", multiline=False, optional=True,
                    tooltip="The small top line above the title. Empty hides it and the accent "
                            "dot. Colour it with wordmark_color.",
                ),
                io.Int.Input(
                    "wordmark_size", default=34, min=8, max=200, optional=True,
                    tooltip="Point size of the wordmark line.",
                ),
                io.String.Input(
                    "bullets", default="", multiline=True, optional=True,
                    tooltip="One bullet per line. First 8 are drawn.",
                ),
                io.String.Input(
                    "chips", default="", multiline=False, optional=True,
                    tooltip="Comma-separated tag chips. First 6 are drawn.",
                ),
                io.String.Input("footnote", default="", multiline=False, optional=True),
                io.String.Input(
                    "footer", default=thumbnail.FOOTER, multiline=False, optional=True,
                    tooltip="Bottom bar text. Empty to omit.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Use THIS as the background instead of drawing the graph — a "
                            "screenshot, a render, anything. Wins over the graph.",
                ),
                io.Boolean.Input(
                    "draw_graph", default=True, optional=True,
                    tooltip="Auto-draw the current workflow as the background. STILL draws when "
                            "an image is connected -- use image_blend to mix them. False = no graph.",
                ),
                io.Float.Input(
                    "image_blend", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="How much of the connected image to show over the graph. "
                            "1.0 = image only, 0.0 = graph only, 0.5 = both. Ignored with no image.",
                ),
                io.Boolean.Input(
                    "stamp_version", default=True, optional=True,
                    tooltip="Append the workflow name/version to the subtitle line.",
                ),
                io.String.Input(
                    "workflow_name", default="", optional=True,
                    tooltip="Label to stamp. Empty = taken from the graph if it carries one.",
                ),
                io.String.Input(
                    "workflow_version", default="", optional=True,
                    tooltip="Version to stamp alongside the name, e.g. 'v16'.",
                ),
                io.Combo.Input(
                    "format", options=["png", "jpg"], default="png", optional=True,
                    tooltip="png embeds the workflow (openable in ComfyUI). jpg cannot carry "
                            "text chunks — card only, no embedded graph.",
                ),
                io.Boolean.Input(
                    "embed_workflow", default=True, optional=True,
                    tooltip="Write the workflow + prompt into the PNG text chunks, exactly as "
                            "ComfyUI's SaveImage does. PNG only.",
                ),
                io.Int.Input("width", default=thumbnail.W, min=256, max=8192, optional=True),
                io.Int.Input("height", default=thumbnail.H, min=256, max=8192, optional=True),
                io.Int.Input("quality", default=88, min=1, max=100, optional=True,
                             tooltip="JPEG quality. Ignored for png."),
                io.String.Input("accent", default="#aa78ff", optional=True,
                                tooltip="Accent colour - top bar, bullets, the dot, BD node titles. "
                                        "Hex '#aa78ff' OR r,g,b '170,120,255' -- both work. ComfyUI has no colour-picker widget for custom nodes, so this is a text field. Clear it to fall back to the default."),
                io.String.Input("wordmark_color", default="#16a34a", optional=True,
                                tooltip="Colour of the wordmark line. "
                                        "Hex '#aa78ff' OR r,g,b '170,120,255' -- both work. ComfyUI has no colour-picker widget for custom nodes, so this is a text field. Clear it to fall back to the default."),
                io.String.Input("background_color", default="#18181e", optional=True,
                                tooltip="Card background. "
                                        "Hex '#aa78ff' OR r,g,b '170,120,255' -- both work. ComfyUI has no colour-picker widget for custom nodes, so this is a text field. Clear it to fall back to the default."),
                io.Boolean.Input("draw_logo", default=True, optional=True,
                                 tooltip="Composite the brand watermark."),
                io.Image.Input(
                    "logo_image", optional=True,
                    tooltip="Use THIS as the watermark instead of a file on disk. Wins over "
                            "logo_path. Works on hosted ComfyUI, where a path would not exist. "
                            "ComfyUI IMAGE is RGB, so connect logo_mask too or the logo "
                            "composites as an opaque rectangle.",
                ),
                io.Mask.Input(
                    "logo_mask", optional=True,
                    tooltip="Alpha for logo_image -- connect LoadImage's MASK output. Without "
                            "it the logo has no transparency.",
                ),
                io.Boolean.Input(
                    "logo_mask_inverted", default=True, optional=True,
                    tooltip="ComfyUI's MASK is INVERTED alpha: 1 means transparent. LoadImage "
                            "emits it that way, so this defaults to True. Set False only if "
                            "your mask is already straight alpha (1 = opaque), otherwise the "
                            "logo comes out as a blank square.",
                ),
                io.String.Input("logo_path", default="", optional=True,
                                tooltip="Watermark file on disk. Ignored when logo_image is "
                                        "connected. Empty = the repo logo."),
                io.Int.Input("logo_height", default=thumbnail.LOGO_H, min=8, max=1024, optional=True),
                io.Int.Input("logo_margin", default=thumbnail.LOGO_MARGIN, min=0, max=512, optional=True),
                io.Combo.Input(
                    "logo_corner",
                    options=["bottom-right", "bottom-left", "top-right", "top-left"],
                    default="bottom-right", optional=True,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="card",
                                tooltip="The rendered card, for previewing in-graph."),
                io.String.Output(display_name="path",
                                 tooltip="Absolute path of the file written."),
                io.String.Output(display_name="status",
                                 tooltip="What was drawn and whether the workflow could be embedded."),
            ],
            hidden=[io.Hidden.extra_pnginfo, io.Hidden.prompt],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, filename_prefix="workflow_card", title="", subtitle="",
                wordmark="BrainDead", wordmark_size=34, bullets="",
                chips="", footnote="", footer=thumbnail.FOOTER, image=None,
                draw_graph=True, stamp_version=True, workflow_name="", workflow_version="",
                image_blend=1.0, format="png", embed_workflow=True, width=None, height=None, quality=88,
                accent="", wordmark_color="", background_color="",
                draw_logo=True, logo_image=None, logo_mask=None,
                logo_mask_inverted=True,
                logo_path="", logo_height=None, logo_margin=None,
                logo_corner="bottom-right") -> io.NodeOutput:

        extra = cls.hidden.extra_pnginfo if isinstance(cls.hidden.extra_pnginfo, dict) else {}
        wf = extra.get("workflow")
        prompt = cls.hidden.prompt

        notes = []
        if wf is None:
            notes.append("no workflow in extra_pnginfo (client did not send it)")

        name = workflow_name.strip()
        if not name and isinstance(wf, dict):
            # LiteGraph keeps no canonical name; these are where one usually ends up
            name = str(wf.get("extra", {}).get("workflow_name")
                       or wf.get("extra", {}).get("name")
                       or wf.get("name") or "").strip()
        ver = workflow_version.strip()

        label = " ".join(p for p in (name, ver) if p)
        head = title.strip() or name or "Workflow"
        sub = subtitle.strip()
        if stamp_version and label and label.lower() not in sub.lower():
            sub = ("%s  ·  %s" % (sub, label)) if sub else label

        bg_img = _to_pil(image)

        cfg = {
            "title": head,
            "subtitle": sub,
            "bullets": _lines(bullets),
            "chips": _csv(chips),
            "footnote": footnote.strip(),
            "footer": footer,
            "wordmark": wordmark,
            "wordmark_size": int(wordmark_size or 34),
            "width": int(width or thumbnail.W),
            "height": int(height or thumbnail.H),
            "quality": int(quality or 88),
            "accent": _rgb(accent, thumbnail.ACCENT),
            "wordmark_color": _rgb(wordmark_color, thumbnail.WORDMARK),
            "bg": _rgb(background_color, thumbnail.BG),
            "no_logo": not draw_logo,
            "logo_path": _logo_source(logo_image, logo_mask, logo_mask_inverted,
                                      logo_path, notes),
            "logo_height": int(logo_height or thumbnail.LOGO_H),
            "logo_margin": int(logo_margin if logo_margin is not None else thumbnail.LOGO_MARGIN),
            "logo_corner": logo_corner,
        }
        if draw_graph and wf:
            cfg["workflow"] = wf
        else:
            cfg["no_graph"] = True
        if bg_img is not None:
            cfg["background"] = bg_img
            cfg["image_blend"] = float(image_blend)
        if bg_img is not None and draw_graph and wf and 0.0 < float(image_blend) < 1.0:
            notes.append("background: graph + image @ %.2f" % float(image_blend))
        elif bg_img is not None and float(image_blend) > 0.0:
            notes.append("background: supplied image")
        elif draw_graph and wf:
            notes.append("background: auto-drawn graph")
        else:
            notes.append("background: none")

        card = thumbnail.render(cfg)   # PIL RGB, not written yet

        ext = "png" if format == "png" else "jpg"
        path = filename_prefix if os.path.isabs(filename_prefix) \
            else os.path.join(OUTPUT_DIR, filename_prefix)
        if not path.lower().endswith("." + ext):
            path = os.path.splitext(path)[0] + "." + ext
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if ext == "png":
            meta = None
            if embed_workflow:
                meta = PngInfo()
                wrote = []
                if wf is not None:
                    meta.add_text("workflow", json.dumps(wf))
                    wrote.append("workflow")
                if prompt is not None:
                    meta.add_text("prompt", json.dumps(prompt))
                    wrote.append("prompt")
                notes.append("embedded: " + (", ".join(wrote) if wrote
                                             else "NOTHING — image will not open in ComfyUI"))
            card.save(path, "PNG", pnginfo=meta)
        else:
            card.save(path, "JPEG", quality=cfg["quality"])
            if embed_workflow:
                notes.append("jpg cannot carry text chunks — workflow NOT embedded")

        status = "%s  |  %s" % (os.path.basename(path), "; ".join(notes))
        print("[BD SaveWorkflowImage] %s -> %s" % (status, path), flush=True)

        out = torch.from_numpy(np.array(card).astype(np.float32) / 255.0).unsqueeze(0)
        return io.NodeOutput(out, path, status)


SAVE_WORKFLOW_IMAGE_V3_NODES = [BD_SaveWorkflowImage]
SAVE_WORKFLOW_IMAGE_NODES = {"BD_SaveWorkflowImage": BD_SaveWorkflowImage}
SAVE_WORKFLOW_IMAGE_DISPLAY_NAMES = {"BD_SaveWorkflowImage": "BD Save Workflow Image"}
