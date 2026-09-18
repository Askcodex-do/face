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

## Status

Foundation complete: structure, config, logging, full pipeline, CLI, GUI, tests.
Phase 1 complete: video engine with progress, cancellation, format support
(MP4/AVI/MKV), and measured memory discipline. 303 tests passing.
Not done yet: `models/` and `assets/` are empty placeholders; single face only;
no packaging; not yet run on real Windows 8.1 hardware.
