# BD Parts Builder 2 Assemble

Build the engine stack (back parts, then the featureless head plate, then front parts) and measure how far it lands from the source.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `complete_parts` | PARTS_BUNDLE | `BD_PartsBuilder2` `complete_parts`: every item as `<item>_back` + `<item>`, in paint order. |
| `plate` | IMAGE | `BD_PartsBuilder2Plates` `eyeless`. |
| `matte` | MASK | `BD_PartsBuilder2Plates` `matte`. |
| `plate_frame` | STRING (socket) | `BD_PartsBuilder2Plates` `plate_frame`. Required: without it the layers cannot be placed on a re-framed plate. If it says `"ok": false` (the plates failed their frame proof), Assemble builds nothing and says why in its report. |
| `plate_source` | IMAGE | `BD_PartsBuilder2Plates` `plate_source`, which the stack is compared against. |
| `plate_head` | MASK (optional) | `BD_PartsBuilder2Plates` `plate_head`. Not wired: non-white pixels of `plate_source`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `engine_parts` | PARTS_BUNDLE | The engine stack in the plate frame: `<item>_back` layers, `head_plate`, `<item>` layers. `depth_median` = stack order. |
| `reassembled` | IMAGE | The stack composited on white. |
| `difference` | IMAGE | Heat map against `plate_source`: red = difference (x3), faint teal = the head area. |
| `report` | STRING | JSON: `diff_on_head`, `share_of_head_over_40`, the paint order, the plate frame. |

## Why this stack

In the engine the head is a bald, featureless plate: no brows, no eyes, no mouth. Everything
else is layered back on: brows, eyes, mouth, hair, hat. This node builds that stack from the
chain's own products and checks it against the source:

```
top      front parts      <item>       (in paint order, back -> front)
         head_plate       eyeless plate over its matte
bottom   back parts       <item>_back  (a hat's back brim, hair behind the head)
```

The paint order is the one `BD_PartsBuilder2` read off the source: where two items overlap,
the one the source shows is in front. Back parts are the pixels of each complete item that sat
over bare skin in the source, which is why they go under the plate.

If `BD_PartsBuilder2Plates` re-framed the head, every complete layer is moved into the plate
frame with the same crop and scale (`plate_frame`), so parts and plate line up.

## Reading the result

- `diff_on_head` is the mean colour difference on the head, in levels (0-255). On the eight
  proving heads it ranged from about 5 to 19, and one reached 26 because its source had lost its
  sunglasses. A large jump usually means an item was drawn at the wrong size (e.g. a complete mouth
  drawn larger) or a plate failed.
- `share_of_head_over_40` is the share of head pixels more than 40 levels off.
- In `difference`, red shapes that follow an item's outline mean that item's complete layer is
  larger or smaller than the source's. A red wash over skin means the plate's skin tone drifted.

## Export

Wire `engine_parts` into `BD_PartsExport` `parts`. The PSD is stacked by `depth_median`, so it
comes out exactly in engine order: back parts, `head_plate`, front parts, one layer each, named by
tag. Leave `base_image` unwired, because the plate is already a layer.

`engine_parts` is in the plate frame. If the plates were re-framed, this PSD's canvas is the crop
recorded in `plate_frame`, not the source canvas. For a PSD in the source frame that rebuilds the
source, use the other export: `BD_PartsBuilder2` `parts` + `base_image` into `BD_PartsExport`.

## Usage

- Use `eyeless` as `plate`, since it is the engine base. `mouthless` works too if the eyes stay
  painted on the plate, but then the eye layers double up.
- `complete_parts` must come from a `BD_PartsBuilder2` run with `make_complete` on, or the stack
  is only the plate.
- All four `BD_PartsBuilder2Plates` frame outputs (`matte`, `plate_frame`, `plate_source`,
  `plate_head`) must come from the same run as `plate`.
