# BD Compare Images

How far a candidate lands from a reference: the red-zone difference the head chain is judged by.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `reference` | IMAGE | What the candidate should reproduce. |
| `candidate` | IMAGE | The result being judged. With alpha it is laid over white first; a different size is resized to the reference. |
| `region` | MASK (optional) | Where to measure (1 = measure). Not wired: pixels that are not white in either image. |
| `gain` | FLOAT | Red = difference x gain. Default 3: 85 levels off is full red. |
| `threshold` | INT | Levels off that count as "off" in the score. Default 40. |
| `label` | STRING (optional) | Shown on the `side_by_side` panel and in the console line. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `difference` | IMAGE | Red = per-pixel mean \|reference - candidate\| over RGB, x `gain`. The region is tinted teal so its outline shows. |
| `side_by_side` | IMAGE | reference \| candidate \| difference in one panel, the score under it. |
| `score` | STRING | JSON: `diff_on_region` (mean difference, 0 = identical, in 0-255 levels), `share_over_<threshold>` (share of the region more than `threshold` levels off), `region_px`, `max_diff`, `label`. |

## Reading it

- **Black with a teal outline** = the candidate reproduces the reference.
- **Red blobs** = where it does not. Look at *where* they are before the numbers: red on a brim edge
  is the paint order; red over the whole face is a moved or restyled head.
- `diff_on_region` is the mean over the region, so a small region with a large error and a large
  region with scattered noise can score alike. `share_over_<threshold>` separates them.

## Where the template uses it (group ⑨ Compare)

| Comparison | Expect |
|------------|--------|
| Visible-layer PSD composite vs MASTER 1 | 0.0: the visible layers own MASTER 1's own pixels, so they rebuild it exactly. Anything else means a layer was redrawn. |
| Engine stack (`BD_PartsBuilder2Assemble` `reassembled`) vs `plate_source`, region `plate_head` | A few levels: the eyeless plate is a generated image, so skin shading differs slightly. Red on features means a layer is missing or out of order. |

The `side_by_side` panels are saved as `parts_builder_2/compare_visible` and `parts_builder_2/compare_engine`.
It is not specific to the head chain: wire any result that should reproduce a source.
