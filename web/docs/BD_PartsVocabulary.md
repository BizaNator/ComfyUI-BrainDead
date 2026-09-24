# BD Parts Vocabulary

The closed list of items `BD_PartsBuilder2` is allowed to extract, plus the matching true/false question for a vision model.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `vocabulary` | STRING (multiline) | One line per item, four fields separated by pipes: key, words the vision model is asked, noun used in edit prompts, accessory (yes/no). Defaults to the 14-item head list below. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `vocabulary` | STRING | The list, passed through unchanged. Wire into `BD_PartsBuilder2` `vocabulary`. |
| `vlm_prompt` | STRING | A true/false question over every listed item. Wire into `AILab_QwenVL` `custom_prompt`. |

## Why a closed list

Qwen Image 2.1 draws whatever it is asked for, whether or not it is in the picture. Ask it to
"extract the glasses" from a head without glasses and it paints some. A vision model therefore
votes first, and only the items it votes present are ever requested. The vote is limited to this
list, so the vision model cannot invent items either. On the eight heads this was proven on,
the closed Qwen3-VL-8B vote got 123 of 128 answers right.

## Line format

```
hair | hair | hair | no
eyebrows | eyebrows | eyebrows | no
eyes | eyes | eyes | no
mouth | mouth | mouth and lips | no
beard | beard | beard | no
moustache | moustache | moustache | no
glasses | glasses | glasses | yes
sunglasses | sunglasses | sunglasses | yes
hat | hat or cap | hat | yes
helmet | helmet | helmet | yes
headband | headband or bandana | headband or bandana | yes
headphones | headphones | headphones | yes
earrings | earrings | earrings | yes
piercings | piercings | piercings | yes
```

| Field | Meaning |
|-------|---------|
| key | The layer's tag (the PSD layer name). Lower-cased. |
| ask | What the vision model is asked about. `hat or cap` merges two words into one item, so the vote cannot split them. |
| noun | What the Qwen 2.1 edit prompt names ("extract the mouth and lips"). |
| accessory | `yes`: the item comes off in `BD_PartsBuilder2`'s head-without-items edit, and it may get a colour retry. Hair, brows, eyes and mouth are never accessories. |

Blank lines and anything after `#` are ignored. Empty fields fall back to the key.

## Wiring

1. Wire `vlm_prompt` into `AILab_QwenVL` `custom_prompt`, with the same headshot on its `image`
   (model `Qwen3-VL-8B-Instruct`).
2. Wire the QwenVL `RESPONSE` into `BD_PartsBuilder2` `presence`.
3. Wire `vocabulary` into `BD_PartsBuilder2` `vocabulary`, so both nodes read the same list.

`presence` accepts the JSON object the vision model returns, even inside prose or code fences. A
JSON key matches an item by its key, its full ask text, or any single word of the ask
(`cap` matches `hat`). You can also type a plain list of keys by hand: `hair, eyes, mouth, hat`.

## Usage

- Add an item by adding a line, for example `mask | face mask | face mask | yes`. Each item the
  vote finds costs two Qwen 2.1 edits (visible + complete), so keep the list to things that
  actually occur on your heads.
- Keep the ask words plain and visual. The vote question tells the model to answer true only if
  the item is clearly visible.
- Changing the list only changes what can be requested. Nothing outside it is ever asked of 2.1.
