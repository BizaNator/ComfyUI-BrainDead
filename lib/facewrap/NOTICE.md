# Third-party assets — FaceWrap pipeline

## `canonical_face_model.obj`

MediaPipe canonical face model — 468 vertices, 898 triangle faces, per-vertex
UV coordinates.

- **Source:** https://github.com/google-ai-edge/mediapipe
- **File:** `mediapipe/modules/face_geometry/data/canonical_face_model.obj`
- **License:** Apache License 2.0
- **Copyright:** Copyright 2019 The MediaPipe Authors.

Used by `BD_FaceFit` (mesh_source="mediapipe_canonical") as the topology + UV
layout for the face-wrap pipeline. The 468 vertex indices correspond 1:1 with
MediaPipe FaceLandmarker landmarks 0..467 (the extra 10 iris/pupil landmarks at
indices 468..477 from `refine_landmarks=True` are NOT mesh vertices).

Apache-2.0 license text: https://www.apache.org/licenses/LICENSE-2.0

## `ict/ict_head_skin.obj` + `ict/ict_landmarks_68.json`

Derived from ICT-FaceKit's `generic_neutral_mesh.obj`. The original is a full
head; these files are the **head-skin subset** (materials `M_Face` +
`M_BackHead` — face, ears, scalp, back of head), triangulated, with the two
UDIM tiles packed into a single [0,1] UV atlas.

- **Source:** https://github.com/USC-ICT/ICT-FaceKit
- **Original files:** `FaceXModel/generic_neutral_mesh.obj`,
  `FaceXModel/vertex_indices.json`
- **License:** MIT License
- **Copyright:** Copyright (c) 2020 USC Institute for Creative Technologies

Regenerated with `tools/preprocess_ict.py` (see that file for the exact
transformation). Used by `BD_FaceFit` (mesh_source="ict_facekit") — gives
ear / scalp / back-of-head coverage that the face-only canonical mesh lacks.
`ict_landmarks_68.json` holds the 68 iBUG landmark vertex indices remapped
into the skin subset, used to Procrustes-fit the ICT head to MediaPipe's
landmarks per view.

MIT license text: https://opensource.org/license/mit
