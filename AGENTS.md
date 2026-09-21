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
several times inside two near-equal photos.

`cubic_stretch_threshold = 1.5` is **kept**, but the justification recorded above
was weak and has been corrected by the Phase 2.5 audit:

* The scale is confirmed to track the *face*, not the image. Holding the canvas at
  700x700 and varying only the drawn face moved the fitted stretch 0.51 -> 2.04
  and flipped the chosen kernel exactly at the threshold. A whole-image metric
  could not move in that test.
* The cost claim was measured wrong the first time. A warp microbenchmark puts
  cubic at 1.7x-2.9x bilinear, but inside the real pipeline detection dominates
  and cubic costs about **1%** of wall time (12.29s -> 12.44s on an 80 frame
  small-face-to-large-face upscale). Do not price this switch from a microbenchmark.
* The benefit claim was metric dependent. Cubic wins on gradient energy (+24%,
  "sharper") and on natural 1/f image statistics (+3% to +18% MAE, +0.5 to +1.6 dB
  PSNR), but *loses* on reconstruction MAE against hard-edged synthetic faces
  (-6% to -13%). Those fixtures are piecewise flat with hard elliptical edges,
  which is exactly where bicubic overshoot rings, so they are biased against cubic
  and must not be used to lower the threshold.
* Pre-scaling the source then warping was measured and is worse - two resampling
  passes blur more than one.

Net: 1.5 is a conservative, near-free guard against soft upscales and is left as
is. Raising or lowering it is not supported by the evidence either way beyond
"roughly here".

## The refinement bound is a safety net, not a regulator

`max_refine_shift = 0.06` is **kept**. The audit shows it never engages on
ordinary input, so its value is not sensitive:

* Headroom (budget / largest natural displacement) ranges 1.19 to 2.79 over the
  validation set, median 1.48. Nothing is throttled.
* Every shift from 0.04 to 0.12 produces *identical* results, because natural
  displacements (7-15px) stay under the budget. Below that it does truncate: the
  cap saturates on all 16 fixtures at 0.02 and on 9 of 16 at 0.04, and accuracy
  degrades at 0.02 (mean eyeErr 6.57px, against 5.97px at 0.04 and above).
* A deliberate distractor (a dark blob 90px below the mouth) failed to drag the
  mouth even with the bound relaxed. What protects the layout is the **confidence
  gating and the search window**, not the cap. Keep the cap as a backstop.
* The frozen invariant holds exactly: jaw, brows and nose (indices 0-35) are
  bit-identical before and after refinement on all 16 fixtures, and a genuinely
  invisible mouth (`low_mouth_contrast`) moves only points 36-47 while 48-67 stay
  put. Any change that perturbs indices 0-35 is a regression.

Two harness traps worth not repeating:

* A per-point displacement can exceed the budget by ~1e-5 px from float32
  rounding. Compare against the budget with a tolerance, or every fixture looks
  like a violation.
* Fitting a landmark set to *itself* has exactly zero residual, so a "self-fit
  alignment residual" metric can never detect anything. Residuals must be measured
  against drawn ground truth to mean anything.

## Known defect: refinement mis-rotates the eye line on rolled faces

Found by the Phase 2.5 audit, **not fixed** (that phase was validation only). This
is the top candidate for the next phase.

The prior handles roll correctly - it is placed using the eye-line rotation. The
refinement search does not: its rectangle is built from the axis-aligned bounds of
the eye block, so on a rolled face the rectangle is a thin horizontal sliver
(~9px tall, grown to ~13px) while the true pupil sits outside it. Measured pupil
excursions outside the search rectangle: 4.6px at +8 degrees, 5.2px at -9, up to
12px on some fixtures.

When the pupil is outside the rectangle the darkest nearby thing wins, which is the
eyebrow. The eye block is then dragged up and sideways, and because the two eyes
fail differently the eye line picks up a *tilt*: with the true roll at -8 degrees
the refined eye line measures +7.3 degrees, i.e. tilted the wrong way.

Consequences, all bounded but real:

* Eye localisation still improves on these fixtures, but far less than on frontal
  ones (10.15px -> 9.33px under roll, against 7.99px -> 2.00px on the frontal case).
* Ground-truth residual gets 3-5x *worse* than the unrefined prior on rolled
  faces (2.90 -> 9.15 and 2.02 -> 9.73). These are the only fixtures where
  refinement is a net loss.
* It does not break conversions: 10/10 frames swap with 0 errors at 0, +/-8, -9
  and 16 degrees. It degrades paste placement, not success rate.

Any fix should make the search rectangle follow the eye-line rotation rather than
the axis-aligned bounds, and should keep the 0-35 frozen and confidence-gated
properties intact.

## Refinement adds about a quarter more inter-frame jitter

At the production smoothing of 0.6, mean frame-to-frame landmark movement on a
drifting 120 frame clip is 0.812px with refinement off and 1.006px with it on
(+24%; p95 1.54 -> 1.80px). Sub-pixel in absolute terms, so accepted, but it is
the real cost of refinement and would show up first on a larger face, where the
same fraction is more pixels.

## Status

Foundation complete: structure, config, logging, full pipeline, CLI, GUI, tests.
Phase 1 complete: video engine with progress, cancellation, format support
(MP4/AVI/MKV), and measured memory discipline. Phase 2 complete: corrected
landmark prior, bounded non-inert refinement, resolution-aware warping.
Phase 2.5 complete (validation/audit only): calibration frozen, wider face-set
results recorded above, one bounded defect documented. 353 tests passing. A 400
frame 640x360 clip converts at ~94 fps with 0 errors, peaking near 100MB.
Not done yet: `models/` and `assets/` are empty placeholders; single face only;
no packaging; not yet run on real Windows 8.1 hardware.
