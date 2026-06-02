# BD MP Face Mask

Extracts per-region face masks using MediaPipe Face Mesh landmarks. Deterministic, CPU-only, ~5 ms per frame. No SAM3 prompts, no sampling noise.

All mask generation is shared with **BD MP Face Export** and **BD MP Face Infill** via `face_mp_shared.py` — masks from all three nodes are pixel-identical for the same landmarks.

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
| `left_iris` | `FACEMESH_LEFT_IRIS` | Requires 478-point model. |
| `right_iris` | `FACEMESH_RIGHT_IRIS` | |
| `irises` | Union of both | |
| `lips` | `FACEMESH_LIPS` | Full lip area: outer + inner contour, both lips. |
| `nose` | Custom landmark set | Bridge (168→5→4), tip (1, 19, 94), left/right ala. MediaPipe has no named nose region. |
| `left_ear` | Face oval lateral slice near lm 234 | Preauricular region visible in front/3-quarter views. |
| `right_ear` | Face oval lateral slice near lm 454 | |
| `ears` | Union of both | |
| `forehead` | Top 40% of face_oval minus brows | Between brow line and hairline. |
| `hair` | Above face_oval, to image top edge | Bounded laterally by face oval width. |

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `image` | — | IMAGE batch. Each frame processed independently. |
| `head_mask` | — | Optional external head silhouette (e.g. SAM3). When wired, used as the `skin` base instead of `face_oval`. |
| `detection_confidence` | 0.5 | MediaPipe detection threshold. |
| `face_expand` | 0 | Pixels to expand (+) or contract (−) the face oval. |
| `feature_expand` | 4 | Pixels to expand eye/brow/lip masks before subtracting from skin. |
| `iris_expand` | 4 | Pixels to expand iris ring outward to fill the coloured disc. |
| `subtract_nose` | False | When True, nose is also excluded from `skin`. |
| `ear_expand` | 25 | Pixels to expand ear region outward from the face oval edge. |
| `hair_expand` | 20 | Pixels to expand hair region downward into the hairline transition. |
| `bbox_feature` | none | Emit a bounding box JSON string for this region. |
| `bbox_frame` | 0 | Which batch frame to extract the bbox from. |

## Outputs

All 18 region outputs are `MASK` tensors `(B, H, W)` in `[0, 1]`. Plus `status` (STRING) and `bbox_json` (STRING).

## Saving Masks for Reuse

If downstream nodes will process a modified image (mouth closed, eyes removed, delighted, albedo prepped), save the masks before modification using **BD MP Save Face Data**. Reload later with **BD MP Load Face Data** — no MediaPipe re-run needed.

## Skin Pipeline Wiring

```
character render ──→ BD MP Face Mask
                          │
                     face_oval ──→ BD_GLSLBatch u_mask0 (face boundary)
                     skin ──────→ BD_GLSLBatch u_mask1 (paintable skin)
                     eyes ──────→ BD_GLSLBatch u_mask2 (eye socket)
                     lips ──────→ BD_GLSLBatch u_mask3 (lips)
```

## Left / Right Convention

MediaPipe uses the **subject's perspective**: `left_eye` is the subject's left eye, which appears on the **right side** of a standard front-facing image.

## Notes

- All regions use convex hull fill except `face_oval`, which uses the exact ordered MediaPipe polygon path (jaw-shaped, non-convex).
- `feature_expand` dilates before subtraction — `skin` gets a clean edge buffer automatically.
- Processing is CPU-bound: ~5 ms at 1K, ~20 ms at 4K.

## Pairs With

- **BD MP Save Face Data** — persist masks for use on modified images
- **BD MP Face Infill** — fill eye/brow/lip sockets; accepts saved face data via `face_data_path`
- **BD MP Face Refine** — refine masks with SAM3 for pixel-accurate boundaries
- **BD MP Face Export** — export landmark JSON + zone mask PNG for Blender UV calibration
- **BD_GLSLBatch** — primary consumer for skin shader
