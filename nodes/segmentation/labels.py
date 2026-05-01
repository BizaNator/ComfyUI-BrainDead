"""
Human parser class labels and palettes.

FASHN: 18-class scheme from fashn-ai/fashn-human-parser (SegFormer-B4).
ATR:   18-class scheme used by mattmdjaga/segformer_b2_clothes and other ATR/LIP
       variants. Class indices differ from FASHN — keep these tables authoritative
       so the mask-split node can map either backend to the same named outputs.
"""

FASHN_LABELS = [
    "background", "face", "hair", "top", "dress", "skirt", "pants", "belt",
    "bag", "hat", "scarf", "glasses", "arms", "hands", "legs", "feet",
    "torso", "jewelry",
]

ATR_LABELS = [
    "background", "hat", "hair", "sunglasses", "top", "skirt", "pants",
    "dress", "belt", "left_shoe", "right_shoe", "face", "left_leg",
    "right_leg", "left_arm", "right_arm", "bag", "scarf",
]

CLOTHING_GROUPS = {
    "all_clothing": ["top", "dress", "skirt", "pants", "belt", "scarf", "hat", "bag"],
    "upper_body":   ["top", "dress", "scarf"],
    "lower_body":   ["pants", "skirt"],
    "accessories":  ["bag", "belt", "hat", "glasses", "sunglasses", "jewelry"],
    "skin":         ["face", "arms", "hands", "legs", "feet", "torso",
                     "left_arm", "right_arm", "left_leg", "right_leg"],
    "footwear":     ["feet", "left_shoe", "right_shoe"],
}

PALETTE = [
    (0, 0, 0),       (255, 224, 189), (139, 69, 19),   (220, 20, 60),
    (255, 105, 180), (75, 0, 130),    (30, 144, 255),  (160, 82, 45),
    (255, 215, 0),   (139, 0, 139),   (255, 140, 0),   (0, 191, 255),
    (255, 192, 203), (255, 99, 71),   (152, 251, 152), (210, 180, 140),
    (218, 165, 32),  (255, 20, 147),
]


def label_index(labels: list[str], name: str) -> int | None:
    try:
        return labels.index(name)
    except ValueError:
        return None
