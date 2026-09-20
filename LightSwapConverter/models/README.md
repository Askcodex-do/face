# Model directory.

LightSwapConverter intentionally ships without learned models. It is fully
offline, CPU only, and 2 GB RAM friendly, so it uses the Haar cascades that are
already bundled inside the opencv-python wheel
(`cv2.data.haarcascades`) instead of downloading weights.

Optional: drop custom cascade XML files in this folder and point
`DetectionConfig` / `LandmarkExtractor(cascades_dir=...)` at it.
