# BD Depth To Shadow Map

Procedural greyscale shadow map from a depth image. Combines top-down N·L lighting with cavity ambient occlusion to produce geometry-aware shadow that automatically carves into recesses (eye sockets, mouth corners, hollow temples) and lifts upward-facing surfaces (forehead, brow ridge, cheekbones).

Designed for the zombie variant of the multi-tone skin shader — drop it in where you'd otherwise hand-paint a "sunken / shadowed face" overlay.

## What it does

Given a depth map, computes:

1. **Surface normal** via finite-difference of depth (`∂z/∂x`, `∂z/∂y`)
2. **N·L lighting** — dot product of surface normal with a user-set light direction (default top-down)
3. **Cavity AO** — depth Laplacian as concavity detector; concave areas (eye sockets, mouth crease) get darkened
4. **Ambient floor** — clamp result to never go below a minimum

Final formula:
```
shadow = clamp(ambient + light_strength × N·L − ao_strength × cavity, 0, 1)
```

Output is a (1, H, W, 3) greyscale image (R=G=B) ready to wire into `u_image3.R` or use as any multiplicative shadow term downstream.

## Why use this over hand-painted shadows?

| Approach | Authoring cost | Per-character | Geometric accuracy |
|---|---|---|---|
| Hand-painted shadow template | Hours per template; reusable | Works on similar faces, breaks on unusual geometry | Wherever you painted |
| **BD_DepthToShadowMap (this node)** | Zero (procedural) | Adapts to each character's actual depth | Always matches actual face geometry |
| Vertical-gradient overlay | Trivial | Uniform top-bright/bottom-dark | None — no socket carving |

The cavity AO term is what creates the "sunken-eye / hollow-cheek" look — a pure vertical gradient can't do that because it has no idea where the sockets are.

## Inputs

| Input | Type | Default | Purpose |
|---|---|---|---|
| `depth` | IMAGE | — | Depth map. Brighter = closer to camera (Lotus2 convention). If RGB, the R channel is used. |
| `light_x` | FLOAT | 0.0 | Light direction X (image-space). `-1`=left, `+1`=right, `0`=centered. |
| `light_y` | FLOAT | -1.0 | Light direction Y (image-space). **`-1`=TOP (default, top-down lighting)**, `+1`=bottom, `0`=horizontal. |
| `light_strength` | FLOAT | 0.7 | How much N·L contributes. `0`=no top-light effect, `1`=full Lambertian, `>1`=exaggerated. |
| `ao_strength` | FLOAT | 0.6 | **The zombie knob.** How much cavity AO contributes. `0`=disabled, `1`=normal, `>1`=more sunken/exaggerated. |
| `ao_radius` | INT | 8 | Pixel radius for cavity detection (Laplacian sigma). 8–16 catches eye sockets and cheek hollows. Smaller = fine creases only. Larger = broader recesses. |
| `ambient` | FLOAT | 0.35 | Minimum output value (shadow floor). `0.35` = output never goes below 35% grey. Prevents pure-black voids in deep cavities. |
| `depth_blur` | FLOAT | 2.0 | Gaussian pre-blur applied to depth before normal computation. Higher = smoother lighting (less noisy). `0` = raw depth (good if depth is already clean). Use 1–3 for Lotus2 output. |
| `invert_depth` | BOOL | False | Toggle if shadows appear inverted (peaks dark instead of lit). Some depth estimators use the opposite convention (darker = closer). |
| `mask` | MASK | optional | Skin/body mask. Pixels outside get the `ambient` floor value (not computed shadow). Use to prevent the background or non-skin areas from generating shadows. |

## Outputs

| Output | Type | Purpose |
|---|---|---|
| `shadow_map` | IMAGE | The combined shadow map (R=G=B greyscale). `1.0`=fully lit, `ambient`=fully shadowed floor. Wire to `u_image3.R` or any multiplicative shadow input. |
| `normal_map` | IMAGE | Derived surface normal visualized as RGB (R=nx, G=ny, B=nz, encoded `[-1,+1] → [0,1]`). Standard tangent-space convention. Useful for debugging the depth-to-normal step or feeding downstream lighting nodes. |
| `cavity_map` | IMAGE | The AO term in isolation (R=G=B). `1.0`=open surface, `0.0`=deepest cavity. **Wire to a PreviewImage during tuning** to see exactly where AO is darkening. |
| `measured_max_curvature` | FLOAT | Peak Laplacian magnitude — useful for cross-character tuning (lets you spot when one character's depth is much flatter/sharper than another). |

## The math, briefly

### Normal derivation
```python
dz_dy, dz_dx = gradient(depth)
normal = normalize(-dz_dx, -dz_dy, 1)
```
Standard depth → surface normal. `z = 1` convention means the surface points generally toward the camera; deviations come from depth gradient.

### N·L for top-down lighting
```python
L = normalize(light_x, light_y, 0)         # default (0, -1, 0)
NdotL = clamp(normal · L, 0, 1)
```
With default `L = (0, -1, 0)` (image-space: -y is up), upward-facing surfaces (where the normal has negative y) get the brightest contribution. Forehead, brow ridge, top of nose, cheekbones light up.

### Cavity AO via Laplacian
```python
laplacian = ∇²(gaussian_blur(depth, sigma=ao_radius))
cavity = clamp(laplacian / max(|laplacian|), 0, 1)
```
The Laplacian of depth is a local curvature measure. Positive Laplacian = concave (valley, socket, crease); negative = convex (peak, ridge). Negative values clipped to zero — we only darken cavities, never brighten peaks.

`ao_radius` controls the spatial scale of the Laplacian (Gaussian sigma). Small radius catches fine creases (mouth lines, nostril shadows). Large radius catches broad hollows (eye sockets, cheek depressions). 8–12 px is a good middle ground for typical face renders at 1024×1024.

### Combination
```python
shadow = clamp(ambient + light_strength × NdotL − ao_strength × cavity, 0, 1)
```
- `ambient` lifts everything off zero (no pure-black voids unless ao_strength is huge)
- `light_strength × NdotL` brightens upward-facing surfaces
- `ao_strength × cavity` darkens concave recesses
- Clamped to [0, 1]

## Tuning recipes

### Normal "all characters get subtle top lighting" (universal use)
```
light_y = -1.0       # top-down
light_strength = 0.3 # subtle
ao_strength = 0.2    # mild socket carving
ao_radius = 10
ambient = 0.5        # high floor — subtle effect overall
```

### Zombie ("dramatic sunken look")
```
light_y = -1.0
light_strength = 0.3     # less top-lit (zombies aren't well-lit)
ao_strength = 1.0–1.5    # crank cavity AO for sunken sockets
ao_radius = 12
ambient = 0.4
depth_blur = 1.0         # sharper sockets
```

### Horror dramatic ("under-lit / flashlight from below")
```
light_x = 0.0
light_y = +1.0           # light from BELOW
light_strength = 0.9
ao_strength = 0.8
ao_radius = 14
ambient = 0.25           # very dark base
```

### Diagnostic ("show me only the cavity AO")
```
light_strength = 0       # disable N·L
ao_strength = 1.0
ambient = 0.5            # mid-grey floor so AO is visible as darkening from neutral
```
Then look at `cavity_map` output to confirm AO is catching the right features.

## Tuning loop

1. **Wire `cavity_map` to a PreviewImage** during tuning — it shows you exactly where AO is detecting recesses, isolated from the lighting term.
2. Adjust `ao_radius` until eye sockets light up properly in `cavity_map`. Too small = nothing visible. Too large = entire face darkens uniformly.
3. **Wire `normal_map` to a PreviewImage** to confirm depth-to-normal step looks correct. Should look like a standard normal map — face areas in blue/purple/pink, edges/transitions in greenish-red.
4. If `normal_map` is noisy / pixelated, increase `depth_blur` (try 3–5).
5. Tune `light_strength` and `ao_strength` independently. They're additive, so each contributes its own visible band of effect.
6. `ambient` controls "how dark is the darkest area." Raise it if shadows are too crushed.

## Wiring patterns

### Pattern A — Zombie iteration only (batched u_image3.R)

```
Lotus2 depth (single)
  → BD_DepthToShadowMap   (ao_strength=1.2, mask=skin_mask)
  → shadow_map output (greyscale)
  → BD_PackChannels       (R=this, G=normal_highlight, B=normal_dark, A=line)
  → batched: [normal_pack, normal_pack, normal_pack, zombie_pack]
  → BD_GLSLBatch.u_image3
```

Iterations 0–2 use the normal shadow pack; iteration 3 (zombie) gets the BD_DepthToShadowMap output baked in.

### Pattern B — Universal use (single u_image3 for all iterations)

```
Lotus2 depth (single)
  → BD_DepthToShadowMap   (ao_strength=0.4, mask=skin_mask)   # subtle
  → shadow_map
  → BD_PackChannels
  → single u_image3 (not batched)
  → BD_GLSLBatch.u_image3
```

Same shadow map used by all 4 tones. Use `vary_floats` to scale shadow strength per iteration if you want zombie to be more dramatic:
```
u_float16=0.3,0.3,0.3,1.0   # iterations 0,1,2 subtle shadow; iteration 3 (zombie) full
```

### Pattern C — Combine with hand-painted overlays

```
Lotus2 depth → BD_DepthToShadowMap → shadow_map (procedural geometry)
                                              ↓
                                       multiply (or max) with
                                              ↓
hand-painted "decay marks" greyscale → procedural + painted → final R channel
                                              ↓
                                       BD_PackChannels → u_image3
```

Get the best of both: automatic geometric AO from depth, plus painted character-specific decay marks (rotting flesh patches, wound shadows, etc.).

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Shadow inverted (peaks dark, sockets bright) | Depth convention mismatch | Toggle `invert_depth` |
| AO doesn't catch eye sockets | `ao_radius` too small | Try 12–16 |
| Whole face uniformly darkens | `ao_radius` too large OR depth is mostly flat | Reduce radius; check that depth has actual face geometry not just a flat plate |
| Noisy / pixelated normal map | Depth is quantized or noisy | Raise `depth_blur` to 3–5 |
| `measured_max_curvature` very small (< 0.001) | Depth has almost no curvature | Confirm depth source is correct; consider falling back to hand-painted |
| Background gets shadowed too | No mask wired | Wire skin/body mask to `mask` input |

## Pairs With

- **Lotus2** depth estimator (already in BD parts pipeline)
- **BD_PackChannels** — combine shadow_map with highlight/dark/line into the RGBA pack for u_image3
- **BD_GLSLBatch** + skin shader — consumer of u_image3.R as shadow multiplier
- **BD_NormalizeLuma** — optional pre-process if depth has wild dynamic range
