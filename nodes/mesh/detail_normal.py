"""
BD_DetailNormalFromAlbedo - derive a high-frequency detail normal from an albedo/diffuse
atlas and blend it onto a base (geometric) normal map.

Why: a high→low bake captures the mesh's *form*, but a model like Trellis generates smooth
surfaces, so the baked normal is mostly flat — no pores / fabric weave / fold micro-detail. That
detail DOES exist in the diffuse atlas. This node extracts the high-frequency component of the
albedo (so flat color regions don't become fake bumps), turns it into a tangent-space detail
normal, and UDN-blends it onto the base normal. Operates per-texel in UV space, so it's safe on a
UV atlas (no global surface assumptions).

Single responsibility: albedo (+ optional base normal) → detailed tangent-space normal map.
"""

import numpy as np

from comfy_api.latest import io


def _to_np(img):
    arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    return arr[..., :3].astype(np.float32)


def _to_tensor(arr):
    import torch
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32)).unsqueeze(0)


class BD_DetailNormalFromAlbedo(io.ComfyNode):
    """Albedo high-frequency detail → tangent-space normal, UDN-blended onto a base normal."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DetailNormalFromAlbedo",
            display_name="BD Detail Normal From Albedo",
            category="🧠BrainDead/Mesh",
            description="Extract high-frequency detail from a diffuse/albedo atlas → tangent-space "
                        "detail normal, UDN-blended onto a base (geometric) normal. Adds skin/fabric/"
                        "fold micro-detail a smooth high→low bake can't capture. UV-safe (per-texel).",
            inputs=[
                io.Image.Input("albedo", tooltip="Diffuse/albedo atlas (e.g. BD_OVoxelBake 'diffuse')."),
                io.Image.Input("base_normal", optional=True,
                               tooltip="Base/geometric tangent-space normal (e.g. OVoxelBake 'normal'). "
                                       "If omitted, only the albedo detail normal is output."),
                io.Float.Input("detail_strength", default=0.6, min=0.0, max=4.0, step=0.05,
                               tooltip="How strongly the albedo detail tilts the normal. ~0.4–1.0 typical."),
                io.Float.Input("high_pass", default=3.0, min=0.5, max=32.0, step=0.5,
                               tooltip="High-pass radius (px). Larger = only finer detail survives, so large "
                                       "albedo color regions don't become fake bumps."),
                io.Boolean.Input("invert", default=False, optional=True,
                                 tooltip="Invert height (treat dark as raised instead of recessed)."),
            ],
            outputs=[
                io.Image.Output(display_name="normal"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, albedo, base_normal=None, detail_strength=0.6, high_pass=3.0,
                invert=False) -> io.NodeOutput:
        from scipy.ndimage import gaussian_filter, sobel

        alb = _to_np(albedo)
        H, W = alb.shape[:2]
        lum = (0.299 * alb[..., 0] + 0.587 * alb[..., 1] + 0.114 * alb[..., 2])

        # High-pass: keep only fine detail; flat color blocks contribute ~0 → no fake cliffs.
        height = lum - gaussian_filter(lum, sigma=float(high_pass))
        if invert:
            height = -height

        # Tangent-space detail normal from height gradients.
        s = float(detail_strength)
        dx = sobel(height, axis=1) * s
        dy = sobel(height, axis=0) * s
        nx, ny, nz = -dx, -dy, np.ones_like(height)
        ln = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / ln, ny / ln, nz / ln  # detail normal in [-1,1]

        if base_normal is not None:
            base = _to_np(base_normal)
            if base.shape[:2] != (H, W):
                from scipy.ndimage import zoom
                zf = (H / base.shape[0], W / base.shape[1], 1)
                base = zoom(base, zf, order=1)
            bx, by, bz = base[..., 0] * 2 - 1, base[..., 1] * 2 - 1, base[..., 2] * 2 - 1
            # UDN (whiteout) blend: add detail XY to base XY, keep base Z.
            cx, cy, cz = bx + nx, by + ny, bz
            cl = np.sqrt(cx * cx + cy * cy + cz * cz) + 1e-8
            nx, ny, nz = cx / cl, cy / cl, cz / cl
            src = "base+detail (UDN)"
        else:
            src = "detail only"

        out = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
        flat = float((out[..., 2] > 0.97).mean()) * 100.0
        status = f"Detail normal: {src}, strength={detail_strength:g}, high_pass={high_pass:g}px, {flat:.0f}% flat"
        print(f"[BD Detail Normal] {status}")
        return io.NodeOutput(_to_tensor(out), status)


DETAIL_NORMAL_V3_NODES = [BD_DetailNormalFromAlbedo]
DETAIL_NORMAL_NODES = {"BD_DetailNormalFromAlbedo": BD_DetailNormalFromAlbedo}
DETAIL_NORMAL_DISPLAY_NAMES = {"BD_DetailNormalFromAlbedo": "BD Detail Normal From Albedo"}
