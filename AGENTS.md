# AGENTS.md

Repository-specific notes for the LightSwapConverter project.

## What this is

A lightweight, fully offline video face replacement converter.
Target: Windows 8.1 64-bit, Python 3.10.11, CPU only, 2 GB RAM, no network.

## Hard constraints (do not violate)

- Python only. Classical computer vision only.
- Never introduce ONNX, TensorFlow, PyTorch, MediaPipe, dlib or any bundled
  neural face model. These are excluded for size, memory and licensing reasons.
- Never add a network call or a runtime model download.
- Keep the memory profile low: one frame in memory at a time, no bulk buffering.

## Environment notes

- **OpenCV 4.x is required.** OpenCV 5.0 removed `cv2.CascadeClassifier` and the
  bundled `haarcascades/` XML files. The pinned `4.8.1.78` includes both. This
  was verified: with 5.0.0 installed, `load_cascade` raises `DetectorError`.
- Pinned deps verified to have Python 3.10 Windows x64 wheels:
  `numpy==1.24.4` (cp310 win_amd64) and `opencv-python-headless==4.8.1.78`
  (cp37-abi3, so it also works on 3.13).
- Tkinter is stdlib on Windows. Pillow is deliberately NOT a dependency: the
  GUI preview converts frames to PPM bytes and passes them to `tk.PhotoImage`.
- For GUI checks on Linux, `sudo apt-get update && sudo apt-get install -y tk8.6
  xvfb`, then run under `xvfb-run -a`. Without `tk8.6`, importing tkinter fails
  with `libtk8.6.so: cannot open shared object file`.

## Commands

```bash
cd LightSwapConverter
python -m pytest tests -q                       # 251 tests
python main.py --probe input.mp4                # inspect a video
python main.py --cli --face f.jpg --video in.mp4 --output out.mp4
python main.py                                  # GUI
```

## Layout conventions

- `main.py` lives *inside* the package directory (per the project spec), so it
  inserts both the package dir and its parent into `sys.path` to work from any
  working directory.
- Package `__init__.py` files use module-level `__getattr__` for lazy re-exports,
  so importing one stage does not drag in the whole stack.
- Test helpers live in `tests/helpers.py`, fixtures in `tests/conftest.py`.
  `tests/` is a package, so import helpers as `from tests.helpers import ...`,
  never as a bare `from conftest import ...`.

## Design decisions worth knowing

- **Detector preprocessing must not use `cv2.equalizeHist`.** This was measured:
  equalisation amplifies compression noise and dropped detection on compressed
  video frames to 1/6. Without it, 6/6. Do not "improve" this by re-adding it.
- **Landmark refinement is always clamped** to a small fraction of the face box.
  The layout is never re-scaled from a single measurement. Alignment uses the
  *relative* geometry between source and target landmarks, and both are measured
  with the same estimator, so systematic bias cancels. A locally wrong
  measurement would not, which is why clamping matters.
- **`feather_mask` re-imposes the eroded core** after blurring. Blurring alone
  attenuates the interior of small masks, making the pasted face look ghostly.
- **Homography error must use the full 3x3 matrix.** Slicing to `[:2]` skips the
  perspective divide and reports a large error for a perfect fit.
- `VideoFaceProcessor.reset()` intentionally clears a stale cancel flag, so a
  cancelled run does not silently break the next one.
- Audio muxing uses `ffmpeg` only when it is already on `PATH`. A missing ffmpeg
  produces silent output rather than an error. Nothing is downloaded.

## Testing conventions

- The suite builds all its media with OpenCV at test time. No binary fixtures,
  nothing downloaded, no mocks: detection, warping and blending run as real code.
- The synthetic face is a drawing, so tests assert *invariants* (point counts,
  region ordering, bounded refinement, mask shape, transform maths) rather than
  pixel accuracy against a cartoon.
- `tests/helpers.draw_synthetic_face` deliberately includes dark eyes, brows and
  a mouth line, because a Haar cascade will not fire on a flat ellipse.

## Module layout note (Phase 1)

`core/processor.py` is the **video processing engine** and is deliberately free
of faces and models. It owns the frame loop, progress, cancellation and
resource handling, and takes the per-frame work as a `transform(frame, index)`
callable. It returns `None` to pass a frame through unchanged.

`core/swapper.py` holds the face pipeline (`VideoFaceProcessor`, `SourceFace`,
`SwapStats`) and drives the engine through a transform closure. Anything that
needs face swapping imports from `swapper`; anything that only needs video IO
uses `processor`. `ProcessingStats` in `swapper` is an alias of `SwapStats`,
kept for older callers.

Cancellation has to reach the engine, because that is where the loop lives.
`VideoFaceProcessor.cancel()` forwards to the active engine and also records a
`_cancel_requested` flag, which is re-checked after the engine is created so a
cancel arriving during source-face preparation is not lost.

## Memory discipline

Verified by measurement, not assumption: peak RSS is ~65 MB and **flat** across
a 50-frame and a 400-frame 640x480 clip, and ~96 MB for a 300-frame 720p clip.
Buffering that 720p clip would cost ~830 MB, which alone would break the 2 GB
target. Keep it that way: one frame in flight, no frame lists, no queues, and
`ProcessingStats.messages` capped by `MAX_RETAINED_MESSAGES` so a file that
fails on every frame cannot grow the list without bound.

## The landmark prior is calibrated, not invented

`core/landmarks.py` predicts 68 points from the detector's own padded box using a
fixed prior table. That table is **not** hand-drawn: it is the bilateral average
of the two 68 point ground truth annotations that ship with OpenCV's test data
(`opencv_extra`, `david1.pts` / `david2.pts`, Apache-2.0), normalised by
interocular distance and eye-line rotation, then placed into the box frame using
two constants: `_PRIOR_INTEROCULAR_FRACTION = 0.2557` and
`_PRIOR_EYE_CENTRE = (0.5064, 0.3981)`.

Those two constants are the whole calibration. If the detector's box framing ever
changes, they are the numbers to re-derive, and `tests/test_landmark_quality.py`
is what will fail if they go stale.

Two traps this codebase has already fallen into, both worth remembering:

* **A wrong prior hides in the alignment fit.** Source and target are described
  by the same layout, so a systematically wrong shape cancels out of the fitted
  similarity transform. Alignment residual stays near zero while the paste sits
  wrongly on the face. Do not treat a low residual as evidence the layout is
  right; check it against real annotations.
* **A tight prior can starve the refinement search.** The eye region is about 3%
  of the box high. If the pupil search window is larger than the rectangle it is
  given, the search returns nothing and refinement silently becomes inert. Helpers
  `_grow_rect_to_at_least` and `_scale_rect` exist to prevent this; both are load
  bearing.

Refinement only touches what darkness can genuinely locate: the two eye centres
and the mouth. Every measurement returns a confidence that scales the movement,
and `_clamp_to_prior` caps each point's displacement from the prior
independently so the layout's shape is preserved. `max_refine_shift = 0` means
"use the prior unchanged".

## Estimator state is per-sequence

`prepare_source_face()` and `process_frame()` share one `LandmarkEstimator`,
whose `estimate()` keeps `_previous` for temporal smoothing at a weight of 0.6.
The source face must be removed from that history before frames are measured -
`prepare_source_face()` resets it. Without the reset the source (a different
face, usually a different size and position) is blended into the first video
frame and the paste shrinks, which measured at 14%.

## Warp interpolation follows the face scale

`FaceTransformer` chooses its warp interpolation from the alignment matrix's
linear scale, not the ratio of the two image sizes. A face can be stretched
several times inside two near-equal photos. Cubic above
`cubic_stretch_threshold` (1.5) kept about 15% more fine detail than bilinear on
faces stretched 3.4x-8.7x; bilinear is kept below that as it is cheaper and
loses nothing near 1:1. Pre-scaling the source then warping was measured and is
worse - two resampling passes blur more than one.

## Status

Foundation complete: structure, config, logging, full pipeline, CLI, GUI, tests.
Phase 1 complete: video engine with progress, cancellation, format support
(MP4/AVI/MKV), and measured memory discipline. Phase 2 complete: corrected
landmark prior, bounded non-inert refinement, resolution-aware warping. 353 tests
passing. A 400 frame 640x360 clip converts at ~94 fps with 0 errors, peaking near
100MB.
Not done yet: `models/` and `assets/` are empty placeholders; single face only;
no packaging; not yet run on real Windows 8.1 hardware.
