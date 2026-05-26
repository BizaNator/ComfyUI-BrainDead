"""
BD_DepthToShadowMap — derive a procedural greyscale shadow map from a depth image.

Combines two geometric shading terms:
  1. Top-down N·L lighting from depth-derived surface normals
     (lights upward-facing surfaces: forehead, brow ridge, nose top, cheekbones)
  2. Cavity AO from depth Laplacian
     (darkens concave areas: eye sockets, mouth corners, under-chin, hollow temples)

Output is a single-channel-equivalent greyscale image suitable for wiring into
u_image3.R of the skin shader, OR any downstream multiplicative shadow term.

Designed to drop into the zombie variant of the multi-tone pipeline — gives a
geometry-aware "sunken / shadowed" look without hand-painting per character.
Pairs naturally with Lotus2 depth output already present in the parts pipeline.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, gaussian_laplace

from comfy_api.latest import io


def _to_2d_depth(depth_t: torch.Tensor) -> np.ndarray:
    """Reduce input image tensor to a single (H, W) float32 depth array."""
    t = depth_t
    if t.ndim == 4:  # (B, H, W, C)
        t = t[0]
    if t.ndim == 3:  # (H, W, C)
        # Use red channel — Lotus2 depth output is RGB greyscale where all channels equal
        return t[..., 0].cpu().numpy().astype(np.float32)
    if t.ndim == 2:  # (H, W)
        return t.cpu().numpy().astype(np.float32)
    raise ValueError(f"depth must be (H,W), (H,W,C), or (B,H,W,C); got {tuple(t.shape)}")


def _mask_to_2d(mask_t) -> np.ndarray | None:
    if mask_t is None:
        return None
    m = mask_t
    if m.ndim == 4:
        m = m.squeeze(0) if m.shape[0] == 1 else m[..., 0]
    if m.ndim == 3:
        m = m[0] if m.shape[0] == 1 else m[..., 0]
    return m.cpu().numpy().astype(np.float32)


class BD_DepthToShadowMap(io.ComfyNode):
    """Procedural shadow map from depth — top-down N·L lighting + cavity AO."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DepthToShadowMap",
            display_name="BD Depth To Shadow Map",
            category="🧠BrainDead/Segmentation",
            description=(
                "Derive a procedural greyscale shadow map from a depth image. Combines:\n"
                "  • N·L lighting from depth-derived surface normal (default top-down)\n"
                "  • Cavity ambient occlusion from depth Laplacian (eye sockets, mouth crease)\n"
                "  • Ambient floor\n\n"
                "Output is a greyscale image suitable for u_image3.R in the skin shader, "
                "or any downstream node that uses a luma-based shadow multiplier.\n\n"
                "Geometry-aware: automatically carves shadow into recesses (eye sockets, "
                "cheek hollows) and lights upward-facing surfaces (forehead, brow ridge). "
                "Designed for the zombie variant of the multi-tone skin pipeline."
            ),
            inputs=[
                io.Image.Input("depth",
                               tooltip="Depth map image. Brighter = closer to camera. "
                                       "Typically from Lotus2 or a depth estimator. "
                                       "If RGB, the R channel is used."),
                io.Float.Input("light_x", default=0.0, min=-1.0, max=1.0, step=0.05,
                               tooltip="Light direction X (image-space). -1=left, +1=right, 0=centered."),
                io.Float.Input("light_y", default=-1.0, min=-1.0, max=1.0, step=0.05,
                               tooltip="Light direction Y (image-space). -1=TOP (default, top-down lighting), "
                                       "+1=bottom, 0=horizontal."),
                io.Float.Input("light_strength", default=0.7, min=0.0, max=2.0, step=0.05,
                               tooltip="How much N·L contributes to final shadow. "
                                       "0=disable top-light, 1=full Lambertian, >1=exaggerated."),
                io.Float.Input("ao_strength", default=0.6, min=0.0, max=2.0, step=0.05,
                               tooltip="How much cavity AO contributes. 0=disable, 1=normal, "
                                       ">1=exaggerated darkness in recesses (more zombie). "
                                       "This is what creates the sunken-eye / hollow-cheek look."),
                io.Int.Input("ao_radius", default=8, min=1, max=64, step=1,
                             tooltip="Pixel radius for cavity detection (Laplacian sigma). "
                                     "8-16 catches eye sockets / cheek hollows. "
                                     "Smaller = fine creases only. Larger = broad hollows."),
                io.Float.Input("ambient", default=0.35, min=0.0, max=1.0, step=0.05,
                               tooltip="Minimum output value (ambient floor). Prevents pure-black voids. "
                                       "0.35 = output never goes below 35% grey. Higher = brighter overall."),
                io.Float.Input("depth_blur", default=2.0, min=0.0, max=16.0, step=0.5,
                               tooltip="Pre-blur applied to depth before normal computation. "
                                       "Higher = smoother lighting (less noisy). 0 = raw depth. "
                                       "Helps if depth has stair-step quantization or noise."),
                io.Boolean.Input("invert_depth", default=False,
                                 tooltip="ON: treat darker pixels as closer (Lotus2 convention is the opposite). "
                                         "Toggle if shadows appear inverted (sockets lit, peaks dark)."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask. Pixels outside the mask get the ambient floor value "
                                      "(not the computed shadow). Use a skin mask to limit AO to face."),
            ],
            outputs=[
                io.Image.Output(display_name="shadow_map",
                                tooltip="Greyscale shadow map (R=G=B). 1.0=fully lit, 0.0=fully shadowed "
                                        "(clamped by ambient floor). Wire to u_image3.R input "
                                        "(or use as a shadow_mult multiplier in any pipeline)."),
                io.Image.Output(display_name="normal_map",
                                tooltip="Derived surface normal visualized as RGB. R=nx, G=ny, B=nz, "
                                        "encoded [-1,+1] → [0,1] (standard tangent-space normal map). "
                                        "Useful for debugging or downstream lighting nodes."),
                io.Image.Output(display_name="cavity_map",
                                tooltip="Cavity AO term in isolation (R=G=B). 1.0=open surface, "
                                        "0.0=deepest cavity. Useful for visualizing what AO is detecting."),
                io.Float.Output(display_name="measured_max_curvature",
                                tooltip="Peak Laplacian magnitude in the depth — useful for tuning "
                                        "ao_strength across characters with different depth scales."),
            ],
        )

    @classmethod
    def execute(cls, depth, light_x=0.0, light_y=-1.0, light_strength=0.7,
                ao_strength=0.6, ao_radius=8, ambient=0.35, depth_blur=2.0,
                invert_depth=False, mask=None) -> io.NodeOutput:

        # ── Depth → 2D float32 ──
        z = _to_2d_depth(depth)
        H, W = z.shape

        if invert_depth:
            z = 1.0 - z

        # Pre-blur to suppress depth quantization / noise before differentiation
        if depth_blur > 0:
            z = gaussian_filter(z, sigma=float(depth_blur))

        # ── Surface normals via finite differences ──
        # np.gradient returns (dz/dy, dz/dx) in image coordinates (axis 0 = rows = y)
        dz_dy, dz_dx = np.gradient(z)

        # Standard depth → normal: normal = normalize(-dz/dx, -dz/dy, 1)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(nx)
        norm_mag = np.sqrt(nx * nx + ny * ny + nz * nz)
        # Avoid divide-by-zero (flat regions)
        norm_mag = np.maximum(norm_mag, 1e-6)
        nx /= norm_mag
        ny /= norm_mag
        nz /= norm_mag

        # ── Top-down N·L (or whatever direction user specified) ──
        # User provides image-space light direction (x, y); z hard-coded 0
        # so light is parallel to the image plane (e.g. y=-1 = light from above).
        lz = 0.0
        l_mag = float(np.sqrt(light_x * light_x + light_y * light_y + lz * lz))
        if l_mag < 1e-6:
            lx_n, ly_n, lz_n = 0.0, -1.0, 0.0  # default top-down
        else:
            lx_n = light_x / l_mag
            ly_n = light_y / l_mag
            lz_n = lz / l_mag

        NdotL = np.clip(nx * lx_n + ny * ly_n + nz * lz_n, 0.0, 1.0)

        # ── Cavity AO via Laplacian of depth ──
        # gaussian_laplace returns ∇²(gaussian * z). Concave surfaces have POSITIVE
        # Laplacian, convex NEGATIVE. We want concave = dark, so flip sign and
        # clip negative (convex) contributions to zero.
        lap = gaussian_laplace(z, sigma=float(ao_radius))
        # Normalize by max absolute value so output is scale-invariant
        lap_max = float(np.max(np.abs(lap))) if lap.size > 0 else 1.0
        lap_max = max(lap_max, 1e-6)
        cavity_amount = np.clip(lap / lap_max, 0.0, 1.0)  # 0=open, 1=deep cavity

        # ── Combine ──
        # shadow = ambient + light_strength × NdotL − ao_strength × cavity
        # ambient sets the floor, NdotL lifts upward-facing surfaces,
        # cavity AO darkens recesses.
        shadow = ambient + light_strength * NdotL - ao_strength * cavity_amount
        shadow = np.clip(shadow, 0.0, 1.0).astype(np.float32)

        # ── Optional skin mask: outside mask, force ambient ──
        if mask is not None:
            m = _mask_to_2d(mask)
            if m.shape != shadow.shape:
                # Resize mask to depth resolution
                import cv2
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            m = np.clip(m, 0.0, 1.0)
            shadow = shadow * m + ambient * (1.0 - m)
            cavity_amount = cavity_amount * m

        # ── Build IMAGE outputs as (1, H, W, 3) ──
        shadow_rgb = np.stack([shadow, shadow, shadow], axis=-1)
        cavity_rgb = np.stack(
            [cavity_amount, cavity_amount, cavity_amount], axis=-1
        ).astype(np.float32)
        normal_rgb = np.stack([
            ((nx + 1.0) * 0.5).astype(np.float32),
            ((ny + 1.0) * 0.5).astype(np.float32),
            ((nz + 1.0) * 0.5).astype(np.float32),
        ], axis=-1)

        shadow_out = torch.from_numpy(shadow_rgb).unsqueeze(0)
        cavity_out = torch.from_numpy(cavity_rgb).unsqueeze(0)
        normal_out = torch.from_numpy(normal_rgb).unsqueeze(0)

        print(f"[BD_DepthToShadowMap] depth shape=({H},{W}), "
              f"L=({lx_n:+.2f},{ly_n:+.2f},{lz_n:+.2f}), "
              f"light_strength={light_strength:.2f}, ao_strength={ao_strength:.2f}, "
              f"ao_radius={ao_radius}, ambient={ambient:.2f}, "
              f"max_curvature={lap_max:.4f}, "
              f"mask={'on' if mask is not None else 'off'}, "
              f"invert_depth={'on' if invert_depth else 'off'}")

        return io.NodeOutput(shadow_out, normal_out, cavity_out, float(lap_max))


DEPTH_TO_SHADOW_MAP_V3_NODES = [BD_DepthToShadowMap]
DEPTH_TO_SHADOW_MAP_NODES = {"BD_DepthToShadowMap": BD_DepthToShadowMap}
DEPTH_TO_SHADOW_MAP_DISPLAY_NAMES = {"BD_DepthToShadowMap": "BD Depth To Shadow Map"}
