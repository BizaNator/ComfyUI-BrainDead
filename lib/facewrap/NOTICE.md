# Third-party assets — FaceWrap pipeline

## `canonical_face_model.obj`

MediaPipe canonical face model — 468 vertices, 898 triangle faces, per-vertex
UV coordinates.

- **Source:** https://github.com/google-ai-edge/mediapipe
- **File:** `mediapipe/modules/face_geometry/data/canonical_face_model.obj`
- **License:** Apache License 2.0
- **Copyright:** Copyright 2019 The MediaPipe Authors.

Used by `BD_FaceFit` as the topology + UV layout for the face-wrap pipeline.
The 468 vertex indices correspond 1:1 with MediaPipe FaceLandmarker landmarks
0..467 (the extra 10 iris/pupil landmarks at indices 468..477 from
`refine_landmarks=True` are NOT mesh vertices).

Apache-2.0 license text: https://www.apache.org/licenses/LICENSE-2.0
