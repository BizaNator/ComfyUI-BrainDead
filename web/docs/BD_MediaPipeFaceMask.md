# BD Face Mask (MediaPipe)

Extracts per-region face masks using MediaPipe Face Mesh landmarks. Deterministic, CPU-only, ~5 ms per frame. No SAM3 prompts, no sampling noise.

## Why MediaPipe instead of SAM3?

| | SAM3 | MediaPipe |
|---|---|---|
| Approach | Neural, prompt-driven | Geometric, landmark-driven |
| Speed | ~300 ms / GPU | ~5 ms / CPU |
| Head boundary | Blob that varies with prompts | Precise face oval (jaw + hairline) |
| Repeatability | Varies per run | Deterministic — identical every run |
| Neck exclusion | Requires careful negative prompts | Automatic — face oval stops at jaw |
| Feature sub-regions | Requires separate SAM3 passes | All regions from one landmark detection |
| Angle limit | Any angle | ~±45° yaw, ±30° pitch |

**Use MediaPipe** as the primary path for character front-facing and 3/4-view renders. Fall back to SAM3 for extreme side profiles or heavily occluded views.

## Landmark Regions

| Region | MediaPipe Source | Notes |
|--------|-----------------|-------|
| `face_oval` | `FACEMESH_FACE_OVAL` (ordered polygon) | Jaw-to-hairline silhouette. No neck, no shoulders. |
| `skin` | face_oval minus features | Primary skin-painting mask. Feature exclusions are pre-expanded by `feature_expand`. |
| `left_eye` | `FACEMESH_LEFT_EYE` | Subject's left eye. **In a standard front-facing image this appears on the right side.** |
| `right_eye` | `FACEMESH_RIGHT_EYE` | Subject's right eye (image left). |
| `eyes` | Union of both | Combined eye mask. |
| `left_brow` | `FACEMESH_LEFT_EYEBROW` | |
| `right_brow` | `FACEMESH_RIGHT_EYEBROW` | |
| `brows` | Union of both | |
| `lips` | `FACEMESH_LIPS` | Full lip area: outer + inner contour, both lips. |
| `nose` | Custom landmark set | Bridge (168→5→4), tip (1, 19, 94), left/right ala (48, 115, 102… / 278, 344, 331…). MediaPipe does not ship a named nose region. |
| `left_ear` | Face oval lateral slice near lm 234 | Preauricular/temporal region visible in front/3-quarter views. Not the full pinna — MediaPipe face mesh does not cover ear cartilage. |
| `right_ear` | Face oval lateral slice near lm 454 | |
| `ears` | Union of both | |
| `forehead` | Top 40% of face_oval minus brows | Between brow line and hairline. |
| `hair` | Above face_oval, to image top edge | Bounded laterally by face oval width. Use with `BD_ColorExtract` for hair colour analysis. |

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `image` | — | IMAGE batch. Each frame processed independently. |
| `detection_confidence` | 0.5 | MediaPipe detection threshold. Lower catches harder angles. |
| `face_expand` | 0 | Pixels to expand (+) or contract (−) the face oval boundary. +4 helps catch jaw-edge hair strands. |
| `feature_expand` | 4 | Pixels to expand eye/brow/lip masks before subtracting from skin. Covers lash shadow and lip border. |
| `subtract_nose` | False | When True, nose is also excluded from `skin`. Use when nose gets a separate shader pass. |
| `ear_expand` | 25 | Pixels to expand ear region outward from the face oval edge. |
| `hair_expand` | 20 | Pixels to expand hair region downward into the hairline transition. |

## Outputs

All 15 region outputs are `MASK` tensors `(B, H, W)` in `[0, 1]`.

| Output | Use |
|--------|-----|
| `face_oval` | Source-of-truth head silhouette |
| `skin` | Wire to GLSL skin shader as the paintable skin region |
| `left_eye` / `right_eye` / `eyes` | Eye-socket exclusion, or separate eye-white/iris tinting |
| `left_brow` / `right_brow` / `brows` | Brow exclusion or separate brow tinting pass |
| `lips` | Lip exclusion or lip colour pass |
| `nose` | Nose exclusion or nose specularity |
| `left_ear` / `right_ear` / `ears` | Ear region for separate skin tone (ears can differ from face) |
| `forehead` | Forehead highlight / specularity separation |
| `hair` | Hair-colour sampling region (feed to `BD_ColorExtract`) |
| `status` | Detection summary — face count, any failed frames |

## Skin Pipeline Wiring

```
character render ──→ BD_MediaPipeFaceMask
                          │
                     face_oval ────────────────────→ BD_GLSLBatch u_mask0 (face boundary)
                     skin ──────────────────────────→ BD_GLSLBatch u_mask1 (paintable skin)
                     eyes ──────────────────────────→ BD_GLSLBatch u_mask2 (eye socket)
                     lips ──────────────────────────→ BD_GLSLBatch u_mask3 (lips)
                     brows + nose (optional)
```

## Left / Right Convention

MediaPipe uses the **subject's perspective**:
- `left_eye` = the subject's left eye = **right side of the image** in a standard front-facing render
- `right_eye` = the subject's right eye = **left side of the image**

This matches anatomy convention (not camera convention). Use the combined `eyes` output if you just want "both eyes excluded from skin" and don't care which side is which.

## Angle Limits

- **Works well**: front-facing, ±30° yaw, ±20° pitch
- **Degrades**: ±30–45° yaw (ear regions on occluded side become unreliable)
- **Fails**: >45° yaw (profile views — use SAM3 instead)

The `face_oval` stays accurate to ~45° yaw. The `left_ear` / `right_ear` on the far side of a 3/4 view will be empty or small — expected behaviour.

## Notes

- All regions are rasterised via convex hull fill except `face_oval`, which uses the exact ordered MediaPipe polygon path (jaw-shaped, non-convex).
- `feature_expand` dilates BEFORE subtraction, so `skin` automatically gains a clean edge buffer around eye sockets and lips without needing separate erode/dilate nodes.
- `hair` extends to the image top edge — clip it with `face_oval` if you want hair restricted to inside the character silhouette.
- Processing is CPU-bound, not GPU. On a 1K image: ~5 ms. On a 4K image: ~20 ms.

## Pairs With

- **BD_GLSLBatch** — primary consumer: pass `skin`, `eyes`, `lips` as u_mask inputs to the 4-tone skin shader.
- **BD_ColorExtract** — wire `hair` mask to extract dominant hair colour for downstream tint matching.
- **BD_LuminanceMask** — combine skin mask with luminance thresholding for shadow/highlight separation within the skin region.
- **BD_SAM3MultiPrompt** — fallback for profile views where MediaPipe fails.
