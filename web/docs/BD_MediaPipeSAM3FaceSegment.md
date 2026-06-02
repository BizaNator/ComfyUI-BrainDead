# BD MP SAM3 Face Segment

MediaPipe-guided SAM3 face-feature segmentation. Runs MediaPipe **once** to localize each feature, then prompts SAM3 per feature with the MediaPipe **bbox + positive landmark points + sibling-feature negative points** — producing pixel-accurate masks in a single node.

**Why guided SAM3 beats text-prompt + IoU** (`BD SAM3 Multi-Prompt` + `BD MP Face Refine`): every SAM3 call is spatially seeded, so there's no prompt ambiguity, no IoU matching, and no exclusion list. Crucially, a positive point **on** the painted brow makes SAM3 grow to the **whole eyebrow object** — solving the stylized-art offset where MediaPipe's human-trained brow landmarks sit below/inside the painted brow (a problem no landmark-only band or envelope can fix).

The outputs match **BD MP Face Mask** (same names/semantics), so this node is drop-in interchangeable — wire it into **BD MP Save Face Data**, **BD MP Face Refine**, or **BD MP Face Infill** exactly like Face Mask.

## Model

Wire a **comfy-core SAM3 `MODEL`** (the same model `SAM3 Detect` uses):

```
Load Diffusion Model (UNETLoader) → sam3.pt → [model]
```

`sam3.pt` must be visible to UNETLoader, i.e. under `models/diffusion_models/` (symlink it from `models/sam3/sam3.pt`). This path uses `model.model.diffusion_model.forward_segment` (the SAM decoder box+point path) and needs no CLIP/text. comfyui-rmbg's text-only SAM3 is **not** used.

## Inputs

| Input | Description |
|-------|-------------|
| `model` | comfy-core SAM3 model (see above). |
| `image` | Full-color, full-resolution face image. Only `image[0]` is processed. |
| `angle` | `front` / `side_left` / `side_right` — stored for downstream/context. |
| `do_brows` / `do_eyes` / `do_lips` / `do_nose` | Per-feature SAM3 toggle. Disabled features fall back to the MediaPipe zone. |
| `detection_confidence` | MediaPipe detection confidence (0.3 works for stylized renders). |
| `min_face_span` | Tiny-detection guard (see **BD MP Face Export**). 0 disables. |
| `mask_threshold` | SAM3 mask probability cutoff (lower → grow thin masks). |
| `refine_iterations` | SAM decoder refinement passes. **Default 1** — extra passes SHRINK masks on stylized renders. |
| `bleed_guard` | Dilate the MediaPipe zone by N px, then clip SAM3 to it. **Large = trust SAM3** (needed for offset brows); 0 = clip exactly to MediaPipe. |
| `cleanup` | Keep only the connected component(s) a positive seed lands in (drops stray SAM3 chunks). |
| `fill_holes` | Fill interior holes during cleanup (teeth/open mouth → solid lips). Default on; turn off for lip-flesh-only. |
| `edge_smooth` | Morphological close+open radius (px @1536) to smooth jagged edges during cleanup. |
| `edge_refine` | Snap the edge to image color/edges: `off` / `guided` (cv2 guidedFilter, fast) / `matting` (PyMatting closed-form, CPU) / `vitmatte` (VitMatte deep model, best soft/hair edges, GPU). Runs on the feature ROI crop. |
| `refine_radius` | Guided-filter radius / matting trimap band width (px @1536). |
| `refine_eps` | Guided-filter edge sensitivity (smaller hugs edges harder). Ignored by matting/vitmatte. |
| `refine_threshold` | Binarize the refined alpha at this level. |

## Outputs

| Output | Description |
|--------|-------------|
| `rgba` | RGBA zone map: **R=lips, G=brows, B=eyes, A=face_oval**. |
| `face_oval`, `skin`, `left_eye`, `right_eye`, `eyes`, `left_brow`, `right_brow`, `brows`, `lips`, `nose`, `irises` | Individual region masks (same names as **BD MP Face Mask**). |
| `status` | Run summary. |

## Where it fits

- **Pixel side:** this node → RGBA + masks (pixel-accurate zones). Wire masks into **BD MP Save Face Data** so everything downstream (**Load Face Data**, **Face Infill**, **Face Refine**) reuses the same accurate regions.
- **Shape side:** **BD MP Face Export** still owns the landmark JSON for Blender UV lineup (geometry, not pixels). This node does **not** replace it.

Single responsibility: this node SEGMENTS. Saving to disk / context is left to the save nodes.

## Recommended settings

Start with `cleanup=on`, `edge_smooth=3`, `edge_refine=guided`. Switch brows to `vitmatte` (or `matting`) when soft brow-hair edges matter. Lower `mask_threshold` a touch if a feature reads thin; raise `bleed_guard` if an offset feature gets clipped.
