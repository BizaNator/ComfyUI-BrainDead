# BD Even Light

Even a head's lighting without flattening its facets.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The shaded head. |
| `mask` | MASK (optional) | Head mask (1 = head). Not wired: pixels that differ from the corner colour. |
| `keep_light` | FLOAT | Share of the lighting gradient kept. 0 = perfectly even, 1 = unchanged. Default 0.5. |
| `facet_gain` | FLOAT | Facet contrast multiplier. 1 = unchanged. Default 0.9. |
| `sigma` | FLOAT | Light blur in px at 1024, scaled with the image. It must be larger than a facet. Default 60. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The head with the light evened; outside the mask the image is unchanged. |
| `report` | STRING | JSON: head mean, light and facet spread before and after. |

## How it works

```
light  = the head's luminance, blurred with a mask-normalised Gaussian (sigma)
facets = luminance - light
out    = mean + keep_light x (light - mean) + facet_gain x facets     (colour carried by ratio)
```

The blur is normalised by the blurred mask, so the background does not darken the light at the
silhouette.

## Why

Qwen Image 2.1 reads a cel-shading prompt ("cell shaded with dynamic shadowing") as one hard key
light: a bright side and a shadow side. Prompts that ask it for even light also flatten the facets.
FaceMaker's albedo, and every D1-D4 variant after it, then carries a lit side and a dark side (owner:
"out of balance on the contrast"). Placed after the 2.1 shading pass, this node keeps the pass's facet
contrast and takes most of the light out.

On don_juan_body, `keep_light` 0.5 and `facet_gain` 0.9 brought the albedo to the balance of the
2509-era v21 the owner approved:

| | v21 (2509) | 2.1 pass alone | 2.1 pass + Even Light |
|---|---|---|---|
| light spread | 22.9 | 31.6 | ~18 |
| facet spread | 39.2 | 47.4 | ~42 |
| left - right | -21 | -45 | ~-25 |

It is deterministic: no model and no seed, so the same input always gives the same output.
