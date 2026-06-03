# BD Derive PBR Maps

Heuristic PBR map derivation (roughness, metallic, AO, normal) from a source image and optional depth/normal/auxiliary inputs, with packed ORM/ARM outputs.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Albedo/color source. Used for roughness (Laplacian), metallic (low-sat high-lum), and albedo output. |
| `depth_image` | IMAGE (optional) | Depth map for AO (gradient magnitude) and normal derivation (Sobel). |
| `normal_map` | IMAGE (optional) | Explicit normal map. When wired, replaces Sobel-derived normals. |
| `silhouette_mask` | MASK (optional) | Character silhouette. Used for `albedo_treatment=silhouette_clip`. |
| `aux_shading_alpha` | IMAGE (optional) | Skin-details GLSL shader alpha output (low=highlights, high=shaded). Drives `shading_to_roughness` and `shading_to_ao`. |
| `aux_detail_texture` | IMAGE (optional) | Detail/lineart source. Laplacian edges boost roughness via `detail_to_roughness`. |
| `metallic_zone_mask` | MASK (optional) | Restrict metallic detection to this region (accessories, clothing, etc.). |
| `depth_sharpen` | FLOAT | Unsharp mask strength applied to depth before AO/normal derivation. |
| `shading_to_roughness` | FLOAT | Weight of `aux_shading_alpha` contribution to roughness. |
| `shading_to_ao` | FLOAT | Weight of `aux_shading_alpha` contribution to AO. |
| `detail_to_roughness` | FLOAT | Weight of `aux_detail_texture` Laplacian contribution to roughness. |
| `albedo_treatment` | COMBO | `edge_pad` (Voronoi nearest-neighbor alpha bleed, default), `silhouette_clip` (clip to silhouette), `passthrough` (no modification). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `albedo` | IMAGE | Processed albedo (with `albedo_treatment` applied). |
| `normal` | IMAGE | Normal map (from `normal_map` input or Sobel-derived from `depth_image`). |
| `roughness` | IMAGE | Roughness map (Laplacian magnitude + luminance inverse + aux contributions). |
| `metallic` | IMAGE | Metallic map (low-saturation + high-luminance detection, optionally zoned). |
| `ao` | IMAGE | Ambient occlusion (depth gradient magnitude + aux shading contribution). |
| `packed_orm` | IMAGE | Packed ORM: R=AO, G=roughness, B=metallic. |
| `packed_arm` | IMAGE | Packed ARM: R=AO, G=roughness, B=metallic (same channels, alternate naming convention). |
| `status` | STRING | Summary of which inputs were active and what derivation was applied. |

## Heuristics

| Map | Algorithm |
|-----|-----------|
| Roughness | Laplacian magnitude + (1 − luminance) + `aux_shading_alpha` × shading_to_roughness + `aux_detail_texture` Laplacian × detail_to_roughness |
| Metallic | Pixels with low saturation AND high luminance within `metallic_zone_mask` |
| AO | Depth gradient magnitude + `aux_shading_alpha` × shading_to_ao |
| Normal | Sobel cross-product from `depth_image`, OR passthrough from `normal_map` |

## Usage

- `albedo_treatment=edge_pad` (default) applies Voronoi nearest-neighbor fill to transparent pixels before returning albedo — prevents UV edge halo in game engines. Use this for any character destined for Unreal or Unity import.
- Wire `aux_shading_alpha` from the skin shader's alpha channel to improve roughness in shaded vs highlight zones — highlights get lower roughness (smoother skin), shadows get higher (pore/bump contribution).
- `metallic_zone_mask` prevents the metallic heuristic from firing on fair skin (low-sat, high-lum pixels). Wire the accessories/clothing segmentation mask to restrict detection to the correct zones.
- `packed_orm` and `packed_arm` are bit-identical — use whichever naming convention your engine expects (Unreal uses ORM; Unity and many exporters use ARM).

## Recommended wiring

```
Source image  →  image
DepthAnythingV2  →  depth_image
Metric3D-Normal  →  normal_map (optional)
Skin shader alpha  →  ImageToMask  →  aux_shading_alpha
Manga lineart  →  aux_detail_texture
Accessories seg  →  metallic_zone_mask
Character alpha  →  ImageToMask  →  silhouette_mask
```
