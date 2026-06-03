# BD Parts Compose

Flatten a PARTS_BUNDLE to a single RGBA canvas by compositing parts back-to-front in depth order.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `parts` | PARTS_BUNDLE | Bundle from `BD_PartsBuilder`, `BD_PartsBatchEdit`, or `BD_PartsRefine`. |
| `output_size` | INT | Scale longest canvas dimension to this value in pixels. 0 = use `frame_size` as-is. |
| `trigger` | BOOL | When False, returns a blank canvas instead of compositing. Wire `is_last` from an iterator node to skip early iterations. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | RGB composite of all parts. |
| `alpha` | MASK | Alpha channel of the composite (union of all part alphas). |

## Compositing order

Parts are sorted by `depth_median` descending (larger depth = farther from camera = painted first). This requires `depth_image` to have been wired into `BD_PartsBuilder`. Without depth data, parts composite in bundle insertion order.

Standard alpha compositing formula per pixel:
```
out_a = src_a + dst_a × (1 - src_a)
out_rgb = (src_rgb × src_a + dst_rgb × dst_a × (1 - src_a)) / out_a
```

## Usage

- Set `output_size=4096` to get a hi-res composite without changing the part images themselves. The canvas is scaled from `frame_size`, parts are scaled proportionally.
- Wire `trigger` from `BD_PartsExport.is_last` or a boolean switch when using `BD_PartsCompose` inside a queue iterator — this prevents unnecessary compositing on early passes.
- For just the alpha mask of the composite, route `alpha` into any downstream mask-aware node without needing a separate ImageToMask conversion.
