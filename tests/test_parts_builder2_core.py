"""BD Parts Builder 2 (UEFN-430) - unit tests for the pure image logic in parts_builder2_core.

parts_builder2_core has no ComfyUI imports (numpy / OpenCV / PIL only), so it is loaded directly by
file path - no live ComfyUI env, no models, no files. Every image here is a small synthetic numpy array.

What is pinned, per the proven run_q21_layers.py recipe it ports:
  * the CLOSED vocabulary and the VLM vote parser (fences, "hat or cap", string "false", typed lists)
  * resolve: a faithful edit owns the SOURCE's pixels; a recoloured edit owns nothing; head repair
  * paint order from complete-layer overlap evidence, depth as the fallback
  * back / front split over bare skin, plate re-framing, to_frame padding per image kind
  * the affine ECC frame proof and its chain composition
  * reassembly back -> plate -> front reproducing the source exactly
  * PARTS_BUNDLE helpers (part_info crop, order_depths)
"""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "nodes" / "segmentation" / "parts_builder2_core.py"


def _load_core():
    spec = importlib.util.spec_from_file_location("parts_builder2_core_under_test", CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_core()

ACCESSORIES = {"glasses", "sunglasses", "hat", "helmet", "headband", "headphones", "earrings", "piercings"}


# ── synthetic images ─────────────────────────────────────────────────────────
def _disc(n, cx, cy, r):
    yy, xx = np.mgrid[:n, :n]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _textured_head(n=512, seed=7):
    """White background, a faceted skin head with brows, eyes, nose and mouth - enough gradient structure
    for the affine ECC to lock onto. -> (rgb uint8, head bool)."""
    rng = np.random.default_rng(seed)
    c, r = n // 2, int(n * 0.38)
    head = _disc(n, c, c, r)
    facets = np.zeros((n, n, 3), np.uint8)
    for _ in range(90):                                  # low-poly facets in skin shades
        pts = rng.uniform(c - r, c + r, (3, 2)).astype(np.int32)
        s = int(rng.integers(150, 225))
        cv2.fillPoly(facets, [pts], (s, int(s * 0.8), int(s * 0.65)))
    base = np.full((n, n, 3), (205, 165, 135), np.uint8)
    face = np.ascontiguousarray(np.where(facets.any(-1, keepdims=True), facets, base))
    k = n / 512.0
    for sx in (-1, 1):                                  # brows, eyes (+ iris)
        ex = int(c + sx * 80 * k)
        cv2.rectangle(face, (int(ex - 45 * k), int(c - 95 * k)), (int(ex + 45 * k), int(c - 80 * k)), (60, 35, 20), -1)
        cv2.ellipse(face, (ex, int(c - 45 * k)), (int(35 * k), int(18 * k)), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(face, (ex, int(c - 45 * k)), int(12 * k), (40, 80, 150), -1)
    cv2.fillPoly(face, [np.array([(c, int(c - 20 * k)), (int(c - 22 * k), int(c + 35 * k)),
                                  (int(c + 22 * k), int(c + 35 * k))], np.int32)], (170, 120, 95))
    cv2.ellipse(face, (c, int(c + 90 * k)), (int(60 * k), int(20 * k)), 0, 0, 360, (170, 50, 60), -1)
    cv2.line(face, (int(c - 60 * k), int(c + 90 * k)), (int(c + 60 * k), int(c + 90 * k)), (80, 20, 25),
             max(1, int(3 * k)))
    img = np.full((n, n, 3), 255, np.uint8)
    img[head] = face[head]
    return img, head


def _warp(img, M):
    return cv2.warpAffine(img, np.float32(M), (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _rgba(rgb, mask, alpha=255):
    a = np.where(mask, alpha, 0).astype(np.uint8)
    return np.dstack([np.where(mask[..., None], rgb, 0).astype(np.uint8), a])


# ── vocabulary ───────────────────────────────────────────────────────────────
def test_default_vocabulary_has_fourteen_items_and_the_accessory_flags():
    voc = C.parse_vocabulary(None)
    assert len(voc) == 14
    assert [v["key"] for v in voc] == ["hair", "eyebrows", "eyes", "mouth", "beard", "moustache", "glasses",
                                       "sunglasses", "hat", "helmet", "headband", "headphones", "earrings",
                                       "piercings"]
    assert {v["key"] for v in voc if v["accessory"]} == ACCESSORIES
    by = {v["key"]: v for v in voc}
    assert by["hat"]["ask"] == "hat or cap" and by["hat"]["noun"] == "hat"
    assert by["mouth"]["noun"] == "mouth and lips"
    assert C.parse_vocabulary("") == voc                 # empty text also means the default


def test_custom_vocabulary_skips_comments_and_blank_lines():
    text = ("# key | ask | noun | accessory\n"
            "\n"
            "   \n"
            "Scarf | scarf or shawl | woollen scarf | yes   # worn round the neck\n"
            "    # an indented comment line\n"
            "freckles\n"
            "tattoo | | | accessory\n")
    voc = C.parse_vocabulary(text)
    assert voc == [
        {"key": "scarf", "ask": "scarf or shawl", "noun": "woollen scarf", "accessory": True},
        {"key": "freckles", "ask": "freckles", "noun": "freckles", "accessory": False},
        {"key": "tattoo", "ask": "tattoo", "noun": "tattoo", "accessory": True},
    ]


def test_vlm_prompt_lists_every_ask():
    voc = C.parse_vocabulary(None)
    p = C.vlm_prompt(voc)
    for v in voc:
        assert v["ask"] in p, v["ask"]
    assert "hat or cap" in p and "headband or bandana" in p
    assert "JSON" in p


# ── the vote ─────────────────────────────────────────────────────────────────
VOC = C.parse_vocabulary(None)


def test_parse_vote_reads_fenced_json_inside_prose():
    reply = ("Sure, here is the answer:\n```json\n"
             '{"hair": true, "eyebrows": true, "eyes": true, "mouth": true, "beard": false, "glasses": false}\n'
             "```\nLet me know if you need more.")
    vote = C.parse_vote(reply, VOC)
    assert set(vote) == {v["key"] for v in VOC}
    assert {k for k, on in vote.items() if on} == {"hair", "eyebrows", "eyes", "mouth"}


def test_parse_vote_maps_the_ask_phrases_back_to_their_keys():
    vote = C.parse_vote('{"hat or cap": true, "headband or bandana": true, "helmet": false}', VOC)
    assert vote["hat"] is True
    assert vote["headband"] is True
    assert vote["helmet"] is False
    # a single alternative word of an "x or y" ask also lands on its key
    vote = C.parse_vote('{"cap": true, "bandana": true}', VOC)
    assert vote["hat"] is True and vote["headband"] is True


def test_parse_vote_string_false_stays_false():
    vote = C.parse_vote('{"hair": "false", "eyes": "true", "beard": "no", "glasses": "0", '
                        '"hat or cap": "False", "mouth": null, "moustache": 0}', VOC)
    assert vote["hair"] is False
    assert vote["eyes"] is True
    for k in ("beard", "glasses", "hat", "mouth", "moustache"):
        assert vote[k] is False, k


def test_parse_vote_json_followed_by_prose_with_braces():
    vote = C.parse_vote('{"hair": true, "eyes": true}\n(I left out the {background}.)', VOC)
    assert {k for k, on in vote.items() if on} == {"hair", "eyes"}


def test_parse_vote_ignores_items_outside_the_closed_list():
    vote = C.parse_vote('{"scarf": true, "tiara": true, "hair": true}', VOC)
    assert set(vote) == {v["key"] for v in VOC}
    assert {k for k, on in vote.items() if on} == {"hair"}


def test_parse_vote_accepts_a_typed_list():
    vote = C.parse_vote("hair, eyes, hat", VOC)
    assert {k for k, on in vote.items() if on} == {"hair", "eyes", "hat"}
    vote = C.parse_vote("Hair\nEYES; hat or cap", VOC)
    assert {k for k, on in vote.items() if on} == {"hair", "eyes", "hat"}
    assert not any(C.parse_vote("", VOC).values())
    assert not any(C.parse_vote(None, VOC).values())


# ── resolve ──────────────────────────────────────────────────────────────────
def _resolve_scene():
    n = 48
    rng = np.random.default_rng(3)
    head = _disc(n, 24, 24, 16)
    src = np.full((n, n, 3), 255, np.uint8)
    src[head] = rng.integers(60, 161, (int(head.sum()), 3))
    ra = np.zeros((n, n), bool)
    ra[4:20, 10:38] = True                               # faithful: reaches above the source alpha
    rb = np.zeros((n, n), bool)
    rb[14:30, 16:32] = True                              # recoloured: entirely on the head, overlaps ra
    assert (ra & ~head).any() and not (rb & ~head).any() and (ra & rb).any()
    faithful = _rgba(src, ra)
    recoloured = _rgba(np.clip(src.astype(np.int16) + 90, 0, 255).astype(np.uint8), rb)
    return src, head, ra, rb, faithful, recoloured


def test_resolve_faithful_edit_owns_its_pixels_with_the_source_colour():
    src, head, ra, rb, faithful, recoloured = _resolve_scene()
    R = C.resolve(src, head, {"hair": faithful, "hat": recoloured})
    assert R["tags"] == ["hair", "hat"]
    hair = R["parts"]["hair"]
    assert hair["fidelity"] == 1.0 and hair["hidden_or_restyled"] is False
    vis = hair["visible"]
    assert vis.dtype == np.uint8 and vis.shape == (48, 48, 4)
    assert np.array_equal(vis[..., 3] == 255, ra)        # alpha exactly 255 on the edit, 0 everywhere else
    assert set(np.unique(vis[..., 3]).tolist()) == {0, 255}
    assert np.array_equal(vis[..., :3], src)             # the SOURCE's pixels, not the edit's
    assert np.array_equal(hair["own"], ra)
    assert np.array_equal(R["owner"], ra.astype(np.uint8))   # owner k+1: hair = 1, nothing is 2


def test_resolve_recoloured_edit_owns_nothing_and_is_flagged():
    src, head, ra, rb, faithful, recoloured = _resolve_scene()
    R = C.resolve(src, head, {"hair": faithful, "hat": recoloured})
    hat = R["parts"]["hat"]
    assert hat["fidelity"] == 0.0
    assert hat["hidden_or_restyled"] is True
    assert not hat["own"].any()
    assert not hat["visible"][..., 3].any()
    assert hat["colour_of_item"] is None
    assert np.array_equal(hat["edit_cover"], rb)         # still trusted for WHERE it is
    assert not (R["owner"] == 2).any()
    # what the recoloured edit covered and nobody explains, as a share of the (repaired) head
    want = (rb & ~ra).sum() / R["head"].sum()
    assert R["unexplained"] == pytest.approx(want)


def test_resolve_repairs_the_head_where_an_edit_covers_outside_the_source_alpha():
    src, head, ra, rb, faithful, recoloured = _resolve_scene()
    R = C.resolve(src, head, {"hair": faithful, "hat": recoloured})
    assert np.array_equal(R["head"], head | ra | rb)
    assert (R["head"] & ~head).any()
    # the repaired pixels above the head are owned by the faithful edit
    assert R["parts"]["hair"]["own"][ra & ~head].all()


def test_resolve_overlap_goes_to_the_edit_that_matches_the_source_best():
    src, head, ra, rb, _, _ = _resolve_scene()
    near = _rgba(src, ra)                                                          # diff 0
    off = _rgba(np.clip(src.astype(np.int16) + 12, 0, 255).astype(np.uint8), rb)   # diff 12: still faithful
    R = C.resolve(src, head, {"a": near, "b": off})
    assert R["parts"]["b"]["hidden_or_restyled"] is False
    assert R["parts"]["a"]["own"][ra & rb].all()
    assert np.array_equal(R["parts"]["b"]["own"], rb & ~ra)
    assert not (R["parts"]["a"]["own"] & R["parts"]["b"]["own"]).any()


def test_resolve_with_no_edits_returns_the_head_untouched():
    src, head, *_ = _resolve_scene()
    R = C.resolve(src, head, {})
    assert R["tags"] == [] and R["parts"] == {}
    assert R["head"] is head and not R["owner"].any()


# ── paint order ──────────────────────────────────────────────────────────────
def _hat_hair(tags):
    n = 40
    hat = np.zeros((n, n), bool)
    hat[0:20] = True
    hair = np.zeros((n, n), bool)
    hair[10:35] = True
    complete = {"hat": _rgba(np.full((n, n, 3), 50, np.uint8), hat),
                "hair": _rgba(np.full((n, n, 3), 90, np.uint8), hair)}
    ih, ir = tags.index("hat") + 1, tags.index("hair") + 1
    owner = np.zeros((n, n), np.uint8)
    owner[hair] = ir
    owner[hat] = ih                                      # the source shows the hat where they overlap
    return complete, owner


@pytest.mark.parametrize("tags", [["hat", "hair"], ["hair", "hat"]])
def test_paint_order_the_item_the_source_shows_is_in_front(tags):
    complete, owner = _hat_hair(tags)
    order, evidence = C.paint_order(tags, complete, owner)
    assert order == ["hair", "hat"]                      # back -> front
    assert evidence == {"hat>hair": [400, 0]}


def test_paint_order_overlap_evidence_beats_depth():
    tags = ["hat", "hair"]
    complete, owner = _hat_hair(tags)
    order, _ = C.paint_order(tags, complete, owner, depth_far={"hat": 0.95, "hair": 0.05})
    assert order == ["hair", "hat"]


def test_paint_order_without_overlap_paints_the_farthest_first():
    n = 30
    a = np.zeros((n, n), bool)
    a[:10] = True
    b = np.zeros((n, n), bool)
    b[20:] = True
    complete = {"near": _rgba(np.full((n, n, 3), 40, np.uint8), a),
                "far": _rgba(np.full((n, n, 3), 80, np.uint8), b)}
    owner = np.zeros((n, n), np.uint8)
    owner[a], owner[b] = 1, 2
    order, evidence = C.paint_order(["near", "far"], complete, owner, depth_far={"near": 0.1, "far": 0.9})
    assert order == ["far", "near"] and evidence == {}
    order, _ = C.paint_order(["near", "far"], complete, owner, depth_far={"near": 0.9, "far": 0.1})
    assert order == ["near", "far"]
    order, _ = C.paint_order(["near", "far"], complete, owner)            # equal depth: list order
    assert order == ["near", "far"]


def test_paint_order_ignores_an_overlap_under_min_px():
    tags = ["hat", "hair"]
    complete, owner = _hat_hair(tags)
    owner[10:20] = 0
    owner[10, :20] = 1                                   # 20 shown pixels < min_px 30
    order, evidence = C.paint_order(tags, complete, owner, depth_far={"hat": 0.9, "hair": 0.1})
    assert evidence == {} and order == ["hat", "hair"]


# ── back / front ─────────────────────────────────────────────────────────────
def test_split_back_front_sends_pixels_over_bare_skin_behind_and_conserves_alpha():
    n = 20
    rng = np.random.default_rng(5)
    comp = np.dstack([rng.integers(0, 256, (n, n, 3)), np.zeros((n, n))]).astype(np.uint8)
    comp[0:10, :, 3] = 255
    comp[10:15, :, 3] = 200
    comp[15:20, :, 3] = 100                              # faint: under the 127 cut, never "back"
    before = comp.copy()
    skin0 = np.zeros((n, n), bool)
    skin0[:, :10] = True
    B, F, nback = C.split_back_front(comp, skin0)
    back = (comp[..., 3] > 127) & skin0
    assert nback == int(back.sum()) == 150
    assert np.array_equal(B[..., 3] > 0, back)
    assert not (F[..., 3][back]).any()
    assert np.array_equal(F[..., 3][~back], comp[..., 3][~back])
    assert np.array_equal(B[..., 3].astype(np.int32) + F[..., 3].astype(np.int32), comp[..., 3].astype(np.int32))
    assert np.array_equal(B[..., :3], comp[..., :3]) and np.array_equal(F[..., :3], comp[..., :3])
    assert np.array_equal(comp, before)                  # input untouched


# ── plate frame ──────────────────────────────────────────────────────────────
def _square_matte(n, side, x0, y0):
    m = np.zeros((n, n), np.float32)
    m[y0:y0 + side, x0:x0 + side] = 1.0
    return m


def test_plate_box_reframes_a_small_head_to_94_percent():
    m = _square_matte(100, 57, 21, 20)                   # head is 57% of the frame
    box = C.plate_box(m)
    assert box is not None
    x0, y0, side = box
    assert 57 / side == pytest.approx(0.94, abs=0.01)
    assert side == 61
    assert x0 <= 21 and x0 + side >= 21 + 57 and y0 <= 20 and y0 + side >= 20 + 57
    assert abs(x0 + side / 2.0 - (21 + 77) / 2.0) <= 1 and abs(y0 + side / 2.0 - (20 + 76) / 2.0) <= 1
    # through to_frame the head then fills ~94% of the plate
    f = C.to_frame(m > 0.5, box, 128)
    bb = C.bbox(f)
    assert (bb[2] - bb[0]) / 128.0 == pytest.approx(0.94, abs=0.015)
    assert (bb[3] - bb[1]) / 128.0 == pytest.approx(0.94, abs=0.015)


def test_plate_box_leaves_a_head_that_already_fills_the_frame():
    assert C.plate_box(_square_matte(100, 95, 2, 3)) is None
    assert C.plate_box(_square_matte(100, 90, 5, 5)) is None      # fill_min is inclusive
    assert C.plate_box(np.zeros((100, 100), np.float32)) is None   # empty matte


def test_to_frame_bool_mask_pads_false_outside_the_image():
    m = np.ones((20, 20), bool)
    out = C.to_frame(m, [-10, -10, 40], 40)
    assert out.dtype == bool and out.shape == (40, 40)
    assert out[10:30, 10:30].all()
    inside = np.zeros((40, 40), bool)
    inside[10:30, 10:30] = True
    assert not out[~inside].any()


def test_to_frame_rgb_pads_white_and_rgba_pads_transparent():
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:] = (50, 100, 150)
    out = C.to_frame(rgb, [-10, -10, 40], 40)
    assert out.shape == (40, 40, 3) and out.dtype == np.uint8
    assert (out[:10] == 255).all() and (out[:, 30:] == 255).all()
    assert (out[10:30, 10:30] == (50, 100, 150)).all()
    rgba = np.dstack([rgb, np.full((20, 20), 255, np.uint8)])
    out = C.to_frame(rgba, [-10, -10, 40], 40)
    assert out.shape == (40, 40, 4)
    assert (out[:10] == 0).all() and (out[:, :10] == 0).all()          # colour AND alpha 0
    assert (out[10:30, 10:30] == (50, 100, 150, 255)).all()


def test_to_frame_output_size():
    rgb = np.full((30, 30, 3), 128, np.uint8)
    assert C.to_frame(rgb, [5, 5, 20], 64).shape == (64, 64, 3)
    assert C.to_frame(rgb, [5, 5, 20], 8).shape == (8, 8, 3)
    assert C.to_frame(rgb[..., 0] > 0, [5, 5, 20], 64).shape == (64, 64)
    assert C.to_frame(np.dstack([rgb, rgb[..., 0]]), [0, 0, 30], 17).shape == (17, 17, 4)
    # box None: unchanged at the image's own square size, else padded to a square first
    assert C.to_frame(rgb, None, 30) is rgb
    wide = np.full((10, 20, 3), 7, np.uint8)
    out = C.to_frame(wide, None, 20)
    assert out.shape == (20, 20, 3)
    assert (out[:10] == 7).all() and (out[10:] == 255).all()


# ── frame proof ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def head512():
    return _textured_head(512)


def test_register_identical_images_pass(head512):
    img, head = head512
    r = C.register(img, img.copy(), head)
    assert r["pass"] is True and r.get("identical") is True
    assert r["scale_x"] == r["scale_y"] == 1.0 and r["centre_move_px"] == 0.0


def _scaled(img, s):
    c = img.shape[0] / 2.0
    return _warp(img, [[s, 0, c - s * c], [0, s, c - s * c]])


def test_register_a_20px_shift_fails(head512):
    img, head = head512
    r = C.register(img, _warp(img, [[1, 0, 20], [0, 1, 0]]), head)
    assert r["pass"] is False, r
    # the coarse-to-fine pyramid locks on and MEASURES the shift: 20 px of 512 = 40 px in 1024 units
    assert r["centre_move_px"] == pytest.approx(40.0, abs=1.0), r
    assert r["scale_x"] == pytest.approx(1.0, abs=0.002) and r["ecc"] > 0.9


def test_register_a_half_pixel_scale_change_passes(head512):
    img, head = head512
    s = 1.0 + 0.5 / int(img.shape[0] * 0.38)             # the head's radius grows by half a pixel
    r = C.register(img, _scaled(img, s), head)
    assert r["pass"] is True, r
    assert r["ecc"] > 0.9
    assert r["scale_x"] == pytest.approx(s, abs=0.001) and r["scale_y"] == pytest.approx(s, abs=0.001)
    assert r["centre_move_px"] <= 3.0


def test_register_a_3_percent_scale_change_fails(head512):
    img, head = head512
    r = C.register(img, _scaled(img, 1.03), head)
    assert r["pass"] is False, r
    assert r["scale_x"] == pytest.approx(1.03, abs=0.002)


def test_register_a_featureless_copy_fails(head512):
    # same silhouette, no features (what a plate edit that wiped the face would look like): ECC cannot
    # converge, and that must read as a failed proof, not a pass at the identity warp
    img, head = head512
    flat = np.full(img.shape, 255, np.uint8)
    flat[head] = 200
    r = C.register(img, flat, head)
    assert r["pass"] is False, r


def test_register_needs_enough_region(head512):
    img, _ = head512
    tiny = np.zeros(img.shape[:2], bool)
    tiny[100:120, 100:120] = True
    r = C.register(img, _warp(img, [[1, 0, 1], [0, 1, 0]]), tiny)
    assert r["pass"] is None


def test_unmeasured_is_ecc_failure_or_too_little_region_not_a_measured_miss(head512):
    # Plates retries a stage's frame proof on the visible-item region ONLY when the proof could not be measured
    # (billpage/front, cast run 2026-09-24: cap + dark glasses + full beard left too little head); a measured scale
    # or offset outside the tolerance is a real miss and must never be re-tried on another region
    img, head = head512
    tiny = np.zeros(img.shape[:2], bool)
    tiny[100:120, 100:120] = True
    too_little = C.register(img, _warp(img, [[1, 0, 1], [0, 1, 0]]), tiny)
    flat = np.full(img.shape, 255, np.uint8)
    flat[head] = 200
    no_converge = C.register(img, flat, head)
    shifted = C.register(img, _warp(img, [[1, 0, 20], [0, 1, 0]]), head)
    passed = C.register(img, img.copy(), head)
    assert C.unmeasured(too_little) and C.unmeasured(passed, too_little)
    if "note" in no_converge:                             # this OpenCV build may also measure a flat copy
        assert C.unmeasured(no_converge)
    assert not C.unmeasured(shifted) and shifted["pass"] is False
    assert not C.unmeasured(shifted, too_little)          # one measured miss rules out the re-try
    assert not C.unmeasured(passed)


def test_chain_proof_multiplies_scales_and_adds_moves():
    up = {"scale_x": 1.1, "scale_y": 0.9, "centre_move_px": 1.25}
    this = {"scale_x": 1.1, "scale_y": 1.0, "centre_move_px": 0.5}
    ch = C.chain_proof(up, this)
    assert ch["scale_x"] == pytest.approx(1.21)          # additive would give 1.2
    assert ch["scale_y"] == pytest.approx(0.9)
    assert ch["centre_move_px"] == pytest.approx(1.75)
    assert ch["pass"] is False


def test_chain_proof_two_passing_links_can_fail_together():
    ident = {"scale_x": 1.0, "scale_y": 1.0, "centre_move_px": 0.0}
    up = {"scale_x": 1.008, "scale_y": 1.0, "centre_move_px": 2.0}
    assert C.chain_proof(up, ident)["pass"] is True
    both = C.chain_proof(up, up)
    assert both["scale_x"] == pytest.approx(1.0161, abs=1e-4)
    assert both["centre_move_px"] == pytest.approx(4.0)
    assert both["pass"] is False
    assert C.chain_proof({"scale_x": 1.008, "scale_y": 1.0, "centre_move_px": 0.0},
                         {"scale_x": 0.995, "scale_y": 1.0, "centre_move_px": 0.0})["pass"] is True


# ── reassembly ───────────────────────────────────────────────────────────────
def _reassembly_scene():
    n = 32
    head = _disc(n, 16, 16, 10)
    src = np.full((n, n, 3), 255, np.uint8)
    src[head] = (200, 160, 130)
    eye = np.zeros((n, n), bool)
    eye[13:16, 11:15] = True
    assert (eye <= head).all()
    src[eye] = (30, 30, 60)
    brim = np.zeros((n, n), bool)
    brim[3:10, 8:25] = True                              # a hat's back brim: part on the head, part above it
    assert (brim & head).any() and (brim & ~head).any()
    src[brim & ~head] = (90, 40, 20)
    plate = np.full((n, n, 3), (0, 255, 0), np.uint8)    # garbage off the head: the matte must hide it
    plate[head] = (200, 160, 130)                        # featureless: no eye
    matte = head.astype(np.float32)
    empty = np.zeros((n, n, 4), np.uint8)
    layers = {"hat": (_rgba(np.full((n, n, 3), (90, 40, 20), np.uint8), brim), empty),
              "eye": (empty, _rgba(src, eye))}
    return src, head, eye, brim, plate, matte, layers


def test_reassemble_plate_plus_front_layer_reproduces_the_source():
    src, head, eye, brim, plate, matte, layers = _reassembly_scene()
    owner = eye.astype(np.uint8)
    comp, heat, rep = C.reassemble(plate, matte, layers, ["hat", "eye"], src, head, owner=owner, tags=["eye"])
    assert comp.dtype == np.uint8 and comp.shape == src.shape and heat.shape == src.shape
    assert np.array_equal(comp, src)
    assert rep["diff_on_head"] == 0.0 and rep["share_of_head_over_40"] == 0.0
    assert rep["per_layer"] == {"eye": 0.0, "skin": 0.0}
    assert not heat[..., 0].any()


def test_reassemble_back_layer_is_hidden_under_the_plate_matte():
    src, head, eye, brim, plate, matte, layers = _reassembly_scene()
    comp, _, _ = C.reassemble(plate, matte, layers, ["hat", "eye"], src, head)
    on, off = brim & head, brim & ~head
    assert (comp[on] == (200, 160, 130)).all()           # matte 1: the plate covers the back brim
    assert (comp[off] == (90, 40, 20)).all()             # matte 0: the back brim shows
    assert (comp[~head & ~brim] == 255).all()            # plate garbage never shows off the matte


def test_reassemble_reports_a_missing_front_layer():
    src, head, eye, brim, plate, matte, layers = _reassembly_scene()
    layers["eye"] = (layers["eye"][0], np.zeros_like(layers["eye"][1]))
    _, heat, rep = C.reassemble(plate, matte, layers, ["hat", "eye"], src, head)
    assert rep["diff_on_head"] > 0
    assert heat[..., 0][eye].all() and not heat[..., 0][~eye].any()


# ── PARTS_BUNDLE helpers ─────────────────────────────────────────────────────
def test_part_info_crops_to_the_alpha_bbox():
    rng = np.random.default_rng(9)
    full = rng.integers(0, 256, (40, 50, 4)).astype(np.uint8)
    full[..., 3] = 0
    full[5:15, 7:20, 3] = 255
    full[9, 19, 3] = 1                                   # any alpha > 0 counts
    depth = np.linspace(-0.5, 1.5, 40 * 50, dtype=np.float32).reshape(40, 50)
    info = C.part_info("hat", full, 0.25, depth, role="front", item="hat")
    assert info["xyxy"] == [7, 5, 20, 15]                # x1, y1, x2, y2 - exclusive
    assert info["img"].shape == (10, 13, 4)
    assert np.array_equal(info["img"], full[5:15, 7:20])
    assert info["img"].flags["C_CONTIGUOUS"]
    assert info["tag"] == "hat" and info["depth_median"] == 0.25
    assert isinstance(info["depth_median"], float)
    assert info["role"] == "front" and info["item"] == "hat"
    want = (np.clip(depth[5:15, 7:20], 0, 1) * 255).astype(np.uint8)
    assert info["depth"].dtype == np.uint8 and np.array_equal(info["depth"], want)
    assert "depth" not in C.part_info("hat", full, 0.25)


def test_part_info_empty_layer_is_none():
    assert C.part_info("hat", np.zeros((16, 16, 4), np.uint8), 0.5) is None
    assert C.bbox(np.zeros((4, 4), bool)) is None


def test_order_depths_strictly_decrease_back_to_front():
    order = ["hat_back", "hair", "eyes", "glasses"]
    d = C.order_depths(order)
    vals = [d[t] for t in order]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert all(0.0 < v < 1.0 for v in vals)
    assert vals == [0.875, 0.625, 0.375, 0.125]
    assert C.order_depths(["only"]) == {"only": 0.5}
    assert C.order_depths([]) == {}


# ── small helpers ────────────────────────────────────────────────────────────
def test_colour_name_and_over_white_and_clamp_alpha():
    assert C.colour_name((238, 241, 239)) == "white"
    assert C.colour_name((28, 22, 26)) == "black"
    rgba = np.array([[[0, 0, 0, 0], [0, 0, 0, 255], [100, 100, 100, 128]]], np.uint8)
    assert C.over_white(rgba).tolist() == [[[255, 255, 255], [0, 0, 0], [177, 177, 177]]]
    matte = np.array([[1.0, 0.0, 0.5]], np.float32)
    assert C.over_white(rgba, matte).tolist() == [[[0, 0, 0], [255, 255, 255], [177, 177, 177]]]
    a = np.array([0.01, 0.05, 0.06, 0.9], np.float32)
    assert C.clamp_alpha(a).tolist() == pytest.approx([0.0, 0.0, 0.06, 0.9])


def test_complete_colour_shift():
    src = np.full((20, 20, 3), 100, np.uint8)
    comp = np.dstack([np.full((20, 20, 3), 130, np.uint8), np.full((20, 20), 255, np.uint8)])
    own = np.zeros((20, 20), bool)
    own[:10] = True
    assert C.complete_colour_shift(comp, src, own) == 30.0
    own[:] = False
    own[:5, :10] = True                                  # 50 px: too few to judge
    assert C.complete_colour_shift(comp, src, own) is None


# ── fixes from the UEFN-430 review ───────────────────────────────────────────
def test_resolve_with_no_edits_has_every_key_the_node_reads():
    src = np.full((64, 64, 3), 200, np.uint8)
    head = np.zeros((64, 64), bool)
    head[10:50, 10:50] = True
    R = C.resolve(src, head, {})
    assert R["unexplained"] == 0.0
    assert R["owned_any"].shape == head.shape and not R["owned_any"].any()


def test_chain_proof_refuses_an_unproven_upstream():
    this = {"scale_x": 1.0, "scale_y": 1.0, "centre_move_px": 0.1, "pass": True}
    for up in ({"pass": False, "note": "ECC failed"}, {"pass": None, "note": "too little region"},
               {"pass": True}):
        ch = C.chain_proof(up, this)
        assert ch["pass"] is False and "note" in ch


def test_plate_box_centres_on_pixel_extents_and_keeps_the_whole_head():
    m = np.zeros((1024, 1024), np.float32)
    m[101:701, 101:701] = 1.0                       # 600 x 600 head, extent centre 401.0
    box = C.plate_box(m, 0.9, 0.99)
    x0, y0, side = box
    assert x0 + side / 2.0 == pytest.approx(401.0, abs=0.51)
    kept = C.to_frame(m > 0.5, box, side)
    assert kept.sum() == 600 * 600


def test_register_kernels_scale_with_resolution():
    rng = np.random.default_rng(3)
    img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    big = np.asarray(C.Image.fromarray(img).resize((2048, 2048), C.Image.NEAREST))
    region = np.zeros((2048, 2048), bool)
    region[200:1800, 200:1800] = True
    assert C.register(big, big.copy(), region)["pass"] is True


def test_flat_fill_makes_each_eye_zone_one_skin_tone():
    img = np.full((256, 256, 3), 200, np.uint8)             # skin
    img[100:130, 60:100] = (40, 20, 10)                     # a dark closed lid + lash line
    img[100:130, 156:196] = (60, 30, 15)
    zone = np.zeros((256, 256), bool)
    zone[104:126, 64:96] = True
    zone[104:126, 160:192] = True
    out, tones = C.flat_fill(img, zone, grow=4, ring=(6, 16), feather=0, min_px=50)
    assert len(tones) == 2
    assert all(t["tone"] == [200, 200, 200] for t in tones)  # sampled from the skin ring, not the lid
    assert (out[100:130, 70:90] == 200).all()               # grown zone covers the lash line top and bottom
    assert (out[110:120, 60:100] == 200).all()              # ... and at both corners of the eye
    assert out[5, 5].tolist() == [200, 200, 200]


def test_flat_fill_ignores_specks_and_empty_zones():
    img = np.full((64, 64, 3), 120, np.uint8)
    zone = np.zeros((64, 64), bool)
    zone[10:12, 10:12] = True                               # below min_px
    out, tones = C.flat_fill(img, zone)
    assert tones == [] and (out == img).all()


def test_even_light_removes_the_gradient_and_keeps_the_facets():
    H = W = 256
    x = np.linspace(-60, 60, W)[None, :].repeat(H, 0)         # a left-to-right key light
    facets = np.where(((np.arange(H)[:, None] // 16) + (np.arange(W)[None, :] // 16)) % 2 == 0, 25.0, -25.0)
    Y = np.clip(128 + x + facets, 0, 255)
    img = np.repeat(Y[..., None], 3, -1).astype(np.uint8)
    head = np.ones((H, W), bool)
    out, st = C.even_light(img, head, keep_light=0.0, facet_gain=1.0, sigma=60.0 * 1024 / W / 4)
    o = out[..., 0].astype(np.float32)
    left, right = o[:, 32:96].mean(), o[:, 160:224].mean()
    assert abs(left - right) < 12                           # was ~60 levels apart
    assert o.std() > 18                                     # the checker (facets) survives
    same, _ = C.even_light(img, head, keep_light=1.0, facet_gain=1.0)
    assert np.abs(same.astype(int) - img.astype(int)).max() <= 2


def test_even_light_leaves_the_outside_alone():
    img = np.full((128, 128, 3), 255, np.uint8)
    img[32:96, 32:96] = 90
    head = np.zeros((128, 128), bool)
    head[32:96, 32:96] = True
    out, _ = C.even_light(img, head)
    assert (out[~head] == 255).all()


# ── engine-stack order fixes (cast run 2026-09-24) ───────────────────────────
def test_headwear_goes_in_front_of_hair():
    order, moved = C.headwear_over_hair(["eyes", "hat", "beard", "hair", "mouth", "glasses"])
    assert order == ["eyes", "beard", "hair", "hat", "mouth", "glasses"] and moved == ["hat"]
    assert C.headwear_over_hair(["hair", "hat"]) == (["hair", "hat"], [])
    assert C.headwear_over_hair(["hat", "eyes"]) == (["hat", "eyes"], [])      # no hair: nothing to reorder


def test_visible_wins_adds_own_pixels_and_clears_over_a_lower_visible_part():
    src = np.zeros((40, 40, 3), np.uint8)
    src[..., 0] = 200
    front = np.zeros((40, 40, 4), np.uint8)
    front[0:20, :, :3] = 50                               # complete layer: garbage over the top half
    front[0:20, :, 3] = 255
    own = np.zeros((40, 40), bool)
    own[25:30, 5:10] = True                               # where the source really shows this item
    lower = np.zeros((40, 40), bool)
    lower[0:10, :] = True                                 # a lower part the source shows here
    F, st = C.visible_wins(front, own, [lower], src, fringe=2)
    assert (F[own, 3] == 255).all() and (F[own, 0] == 200).all()     # own pixels: source colour, opaque
    assert st["visible_added_px"] == 25
    assert (F[0:12, :, 3] == 0).all()                     # cleared over the lower part + 2 px fringe
    assert (F[13:20, :, 3] == 255).all()                  # hidden completion elsewhere is kept
    assert st["clipped_px"] == 12 * 40
