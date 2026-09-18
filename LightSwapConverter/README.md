# LightSwapConverter

A lightweight, fully offline video face replacement converter.

This repository currently contains the **project foundation**: the package
structure, the configuration and logging utilities, a complete processing
pipeline built from classical computer vision, a Tkinter interface and a test
suite. See [Status](#status) for what is finished and what is deliberately left
for later.

## Target environment

| Constraint | Value |
| --- | --- |
| Operating system | Windows 8.1 64-bit |
| Python | 3.10.11 |
| Memory | 2 GB RAM |
| Compute | CPU only, no GPU |
| Network | Never used. The application is fully offline |

Every default in `utils/config.py` is chosen for that machine: two OpenCV
worker threads, no seamless blending by default, one frame in memory at a time.

## Design constraints

The project uses classical computer vision only. The following are explicitly
**not** used, and are not dependencies of this project:

- ONNX / `onnxruntime`
- TensorFlow
- PyTorch
- MediaPipe, dlib, or any other bundled neural face model
- Any cloud or network service

Face detection uses OpenCV's bundled Haar cascades. Landmarks are estimated
geometrically from the detection box plus local image measurements. Alignment is
a least squares similarity fit. Blending is a feathered alpha composite. All of
this runs on a CPU in a few megabytes of memory, which is what the target
machine can afford.

## Requirements

```
numpy==1.24.4
opencv-python-headless==4.8.1.78
```

Tkinter ships with CPython on Windows and is not listed. Pillow is deliberately
absent: the preview converts frames to PPM and hands the bytes straight to
Tkinter.

> **OpenCV 4.x is required.** OpenCV 5.0 removed `CascadeClassifier` and the
> bundled cascade XML files. The pinned 4.8.1 release includes both. If you
> upgrade OpenCV, detection will fail with a clear `DetectorError` rather than
> silently doing nothing.

Install:

```
python -m pip install -r requirements.txt
```

## Layout

```
LightSwapConverter/
    __init__.py
    main.py                 entry point, GUI and CLI
    gui/
        __init__.py
        window.py           Tkinter interface
    core/
        __init__.py
        video_reader.py     frame accurate decoding, metadata, seeking
        video_writer.py     encoding, optional audio muxing via ffmpeg
        face_detector.py    Haar cascade detection with tracking and fallbacks
        landmarks.py        geometric 68 point estimator
        alignment.py        similarity, affine and homography fits
        transformer.py      warping, colour transfer, sharpening
        blender.py          feathered alpha and seamless compositing
        processor.py        video processing engine (frame loop, progress, cancel)
        swapper.py          face pipeline orchestration over the engine
    utils/
        __init__.py
        config.py           dataclass configuration, JSON persistence
        logger.py           rotating file log plus in-memory buffer for the GUI
    models/                 reserved for future data files (empty)
    assets/                 icons and other static files (empty)
    tests/                  pytest suite
    requirements.txt
    README.md
```

## Usage

Graphical interface:

```
python main.py
```

Headless conversion, useful for batch jobs and for checking a machine:

```
python main.py --cli --face face.jpg --video input.mp4 --output output.mp4
```

Inspect a video without processing it:

```
python main.py --probe input.mp4
```

Useful options:

```
--resize 640x360        process at a lower resolution, much faster
--max-frames 100        stop after 100 frames
--no-audio              do not copy the source audio track
--seamless              gradient aware blending, slower
--log-level DEBUG       verbose logging
--config path.json      use a specific configuration file
```

Settings are saved to `%APPDATA%\LightSwapConverter\config.json` on Windows, and
log output goes to the same directory.

### Audio

OpenCV writes video only. When `ffmpeg` happens to be on `PATH`, the source
audio track is muxed into the finished file. If `ffmpeg` is missing the output
is still produced, without audio. Nothing is ever downloaded.

## How the pipeline works

The stages are separate modules with narrow interfaces, so each one can be
tested and replaced independently.

1. **`video_reader`** decodes one frame at a time. It reports size, frame rate
   and frame count, and supports seeking. Frames are never buffered in bulk.
2. **`face_detector`** finds the most prominent face. It searches a region
   around the previous detection first, which is both faster and more stable on
   video, then falls back to a full frame search and to alternative cascades.
   Histogram equalisation is deliberately not applied, because it amplifies
   compression noise and measurably lowers the detection rate on compressed
   video.
3. **`landmarks`** estimates a 68 point layout. A proportional prior places
   every point, then the eyes, nose and mouth are nudged towards image evidence
   by searching for the darkest patch in a region derived from the prior. Every
   nudge is clamped to a small fraction of the face box, so a false measurement
   cannot drag the layout off the face. Positions are smoothed across frames to
   remove jitter.
4. **`alignment`** fits a similarity transform from the source face to the
   target face using the two eye centres and the nose tip. Similarity preserves
   shape and cannot introduce shear. Affine and homography fits are available
   for stronger head poses. A fit whose reprojection error is too large is
   rejected, and the frame is passed through untouched rather than smeared.
5. **`transformer`** warps the source face into the target face box and matches
   its colour statistics to the target region in LAB space, with the contrast
   correction clamped so an unusual region cannot blow up the result.
6. **`blender`** builds a feathered alpha channel constrained to the convex hull
   of the target landmarks, so the corners of the warped rectangle never show.
   The eroded core of the mask is kept fully opaque, which stops small faces
   from looking ghostly.
7. **`video_writer`** encodes the result, then optionally muxes the audio.

`processor` wires these together and reports progress, preview frames and
cancellation back to the caller.

### An honest note on landmark accuracy

The landmark estimator is a geometric estimator, not an anatomical face model.
It cannot be as accurate as a neural network, and it does not attempt to be.

What it relies on is *relative* consistency. Alignment maps source landmarks
onto target landmarks, and both are measured with the same estimator using the
same proportional prior. Any systematic bias affects both sides and cancels out
in the fitted transform. A locally wrong measurement would not cancel, which is
why every refinement is tightly clamped and why the layout is never re-scaled
from a single measurement.

Expect good results on frontal faces and reasonable results on moderate
rotation. Strong profile views are out of scope for a detector of this class.

## Tests

```
python -m pytest tests -q
```

The suite builds its own media. Videos and images are drawn with OpenCV at test
time, so no sample files are needed and nothing is downloaded. Detection,
warping and blending are exercised as real code paths; no stage is mocked.

251 tests currently pass, covering configuration round trips and validation,
log handler behaviour, video reading and writing, detection, landmark
invariants, transform mathematics, warping, colour transfer, blending, the
orchestration loop, progress reporting and cancellation.

## Status

Implemented and tested:

- Full project structure, configuration and logging
- Frame accurate video reading and writing, with optional audio pass-through
- Face detection, landmark estimation, alignment, warping, blending
- End to end conversion, verified on synthetic video at 10/10 frames swapped
- Headless CLI and the Tkinter interface
- Test suite

Left for later, as agreed for this phase:

- **`models/` is empty.** No model files are shipped. The directory exists so a
  future detector can drop its data in without a layout change.
- **`assets/` is empty.** `gui/window.py` already loads `assets/icon.png` when
  it is present, so adding an icon needs no code change.
- **Single face only.** `FaceDetector.detect_all` returns every face found, but
  the processor replaces one. Multi-face selection is a natural next step.
- **No packaging.** There is no PyInstaller spec or installer yet.
- **Windows 8.1 has not been tested directly.** Development and verification
  happened on Linux with Python 3.13. The pinned dependency versions are chosen
  for Python 3.10 on Windows, and the code avoids anything platform specific,
  but a run on the real target machine is still outstanding.
