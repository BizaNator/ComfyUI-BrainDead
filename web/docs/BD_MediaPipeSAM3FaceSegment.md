# BD MP SAM3 Face Segment

MediaPipe-guided SAM3 face-feature segmentation. Runs MediaPipe **once** to localize each feature, then prompts SAM3 per feature with the MediaPipe **bbox + positive landmark points + sibling-feature negative points** — producing pixel-accurate masks in a single node.

**Why guided SAM3 beats text-prompt + IoU** (`BD SAM3 Multi-Prompt` + `BD MP Face Refine`): every SAM3 call is spatially seeded, so there's no prompt ambiguity, no IoU matching, and no exclusion list. Crucially, a positive point **on** the painted brow makes SAM3 grow to the **whole eyebrow object** — solving the stylized-art offset where MediaPipe's human-trained brow landmarks sit below/inside the painted brow (a problem no landmark-only band or envelope can fix).

The outputs match **BD MP Face Mask** (same names/semantics), so this node is drop-in interchangeable — wire it into **BD MP Save Face Data**, **BD MP Face Refine**, or **BD MP Face Infill** exactly like Face Mask.

## Model

**No wiring needed.** Leave `model` unwired and the node **auto-loads + auto-downloads** the official SAM3 checkpoint (`Comfy-Org/sam3.1`) in-house via `bd_sam3` on first use (into `models/checkpoints/`). It uses `model.model.diffusion_model.forward_segment` (the SAM decoder box+point path; no CLIP/text needed). Standalone — comfyui-rmbg is **not** used.

Wire a comfy-core SAM3 `MODEL` (e.g. `Load Diffusion Model → sam3.pt`) only to override the auto-loaded model.

## Inputs

| Input | Description |
|-------|-------------|
| `model` | comfy-core SAM3 model (see above). |
| `image` | Full-color, full-resolution face image. Only `image[0]` is processed. |
| `angle` | `front` / `side_left` / `side_right` — stored for downstream/context. |
| `do_brows` / `do_eyes` / `do_lips` / `do_nose` | Per-feature SAM3 toggle. Disabled features fall back to the MediaPipe zone. |
| `do_ears` | Segment ears with **text-grounded SAM3** ("ear", split L/R at the face centre) instead of the weak MediaPipe approximation (the 478-pt face mesh has no outer-ear points). Off = MediaPipe ears (now non-empty). |
| `remove_background` | Compute a clean **head silhouette in-house** (text-SAM3): `head_prompts` positive minus `exclude_prompts` (neck/shirt/…) negative → head minus neck/clothing/background. Clips every output to it, becomes `head_mask`, and is emitted as the `silhouette` output. Replaces a separate bg-removal + silhouette chain. Skipped if `silhouette_mask` is wired. |
| `head_prompts` / `exclude_prompts` | Positive / negative word prompts (one per line) for `remove_background`. |
| `neck_cut` | With `remove_background`: drop everything below the MediaPipe **jawline** (`face_oval`'s lowest edge per column — follows the chin/jaw contour, *not* a flat line; hair/ears to the sides are kept). This is how the neck is removed: `head` (kept for the cranium/bald crown) pulls the neck in, and SAM3 can't reliably segment the bare neck as a text negative, but MediaPipe's jaw landmarks are accurate so the cut tracks the jaw exactly. |
| `detection_confidence` | MediaPipe detection confidence (0.3 works for stylized renders). |
| `min_face_span` | Tiny-detection guard (see **BD MP Face Export**). 0 disables. |
| `mask_threshold` | SAM3 mask probability cutoff (lower → grow thin masks). |
| `refine_iterations` | SAM decoder refinement passes. Sharpens brows/eyes; **lips always use a single pass** (the loop destabilizes them) and a collapse-guard reverts any feature that the loop over-shrinks. |
| `bleed_guard` | Dilate the MediaPipe zone by N px, then clip SAM3 to it, for **brows/eyes/nose**. **Large = trust SAM3** (brows need ~40-45 to grow out and fill); 0 = clip exactly to MediaPipe. |
| `lips_bleed_guard` | Same, but for **lips only** (default 12). Kept separate because brows need a *large* guard to fill while lips need a *small* one or they overflow onto the face. Lower = tighter lips; 0 = clip exactly to the MediaPipe lip contour. |
| `eye_bleed_guard` | Same, but for **eyes only** (default 10). SAM3 grows to the whole eye (lid/lashes/sclera); a small guard clips it back toward the eyelid aperture so the mask hugs the eyeball — like **BD Face Infill**'s eroded eyelid hull. Lower = tighter; 0 = clip exactly to the MediaPipe eye contour. |
| `cleanup` | Keep only the connected component(s) a positive seed lands in (drops stray SAM3 chunks). |
| `fill_holes` | Fill interior holes on **non-lip** features (eyes/nose/etc.) so each is solid. Lips are controlled by `lips_mode`. |
| `lips_mode` | Lips-specific: `mouth` (default) fills the whole mouth area (lips + teeth + tongue) into the lips mask; `lips_only` keeps lip flesh only (color-aware `edge_refine` excludes teeth/tongue). |
| `edge_smooth` | Morphological close+open radius (px @1536) to smooth jagged edges during cleanup. |
| `edge_refine` | Snap the edge to image color/edges: `off` / `guided` (cv2 guidedFilter, fast) / `matting` (PyMatting closed-form, CPU) / `vitmatte` (VitMatte deep model, best soft/hair edges, GPU). Runs on the feature ROI crop. |
| `refine_radius` | Guided-filter radius / matting trimap band width (px @1536). |
| `refine_eps` | Guided-filter edge sensitivity (smaller hugs edges harder). Ignored by matting/vitmatte. |
| `refine_threshold` | Binarize the refined alpha at this level. |
| `vitmatte_model` | `small` / `base` — **auto-downloaded from the HF hub** (`hustvl/vitmatte-{small,base}-composition-1k`) on first use. Standalone; no other node pack required. |
| `silhouette_mask` | Optional head silhouette (e.g. SAM3 head). When wired, **all outputs are clipped to it**, it becomes `head_mask`, and `masked_skin` = skin within it. |
| `head_mask` | Optional inner head/face-plate mask used as the skin base (`skin = head_mask − eyes − brows − lips`). Falls back to `face_oval`; echoed to the `head_mask` output. |

## Outputs

Full uniform region set — **identical names to BD MP Face Mask**, and a drop-in for the inputs of **BD MP Save Face Data**:

| Output | Description |
|--------|-------------|
| `rgba` | RGBA zone map: **R=lips, G=brows, B=eyes, A=face_oval**. |
| `face_oval`, `skin`, `left_eye`, `right_eye`, `eyes`, `left_brow`, `right_brow`, `brows`, `left_iris`, `right_iris`, `irises`, `lips`, `nose`, `left_ear`, `right_ear`, `ears`, `forehead`, `hair` | Region masks. SAM3-segmented: eyes, brows, lips, nose. MediaPipe-only (no SAM3): iris, ears, forehead, hair. |
| `head_mask` | Resolved head mask (silhouette/head_mask input, else face_oval). |
| `masked_skin` | Skin clipped to the silhouette — wire to **BD MP Save Face Data** `masked_skin`. |
| `debug_overlay` | Render with feature masks tinted (lips=R, brows=G, eyes=B, nose=Y). |
| `status` | Run summary. |

> **Consolidation:** with the full region set + `silhouette_mask`/`head_mask` inputs + `debug_overlay`, this node covers the guided-SAM3 path on its own and feeds **BD MP Save Face Data** directly. **BD MP Face Mask** stays as the no-SAM3-model (fast, landmark-only) option; **BD MP Face Refine** stays for the text-prompt SAM3-batch + IoU workflow.

## Where it fits

- **Pixel side:** this node → RGBA + masks (pixel-accurate zones). Wire masks into **BD MP Save Face Data** so everything downstream (**Load Face Data**, **Face Infill**, **Face Refine**) reuses the same accurate regions.
- **Shape side:** **BD MP Face Export** still owns the landmark JSON for Blender UV lineup (geometry, not pixels). This node does **not** replace it.

Single responsibility: this node SEGMENTS. Saving to disk / context is left to the save nodes.

## Recommended settings

Start with `cleanup=on`, `edge_smooth=3`, `edge_refine=guided`. Switch brows to `vitmatte` (or `matting`) when soft brow-hair edges matter. Lower `mask_threshold` a touch if a feature reads thin; raise `bleed_guard` if an offset feature gets clipped.
