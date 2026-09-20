# LightSwapConverter

A lightweight, fully offline video face replacement converter for low-end
Windows machines.

**Status: project foundation.** The architecture, module interfaces, GUI shell
and end-to-end pipeline scaffolding are in place and tested. The replacement
itself is currently a deterministic image-processing swap (aligned copy +
colour match + feathered blend), not a learned model. See
[Roadmap](#roadmap).

## Target environment

| Constraint | Value |
| --- | --- |
| OS | Windows 8.1 64-bit |
| Python | 3.10.11 (64-bit) |
| RAM | 2 GB |
| Compute | CPU only, no GPU |
| Network | none - fully offline |

## Design rules

- **Python only.** No C/C++ build step, no compiler needed on the target machine.
- **Lightweight libraries.** numpy, OpenCV (`opencv-python`), Pillow for image
  import. Tkinter for the GUI because it ships with CPython on Windows.
- **No ONNX, no TensorFlow, no PyTorch, no large model files.**
- **No cloud services.** No downloads, no telemetry, no API calls.

## Memory strategy

2 GB of RAM is the binding constraint, so the pipeline never holds the video in
memory:

- frames are decoded one at a time and written out as soon as they are processed;
- input is downscaled to at most 960x540 by default (`VideoConfig.max_width` /
  `max_height`);
- only one aligned face patch and one soft mask are alive per frame;
- OpenCV and Tkinter are the only heavy imports, and both are lazily used.

## Project layout

```
LightSwapConverter/
  main.py                 entry point (GUI by default, CLI with --headless)
  gui/
    window.py             Tkinter window, worker thread, progress reporting
  core/
    video_reader.py       sequential decoding + downscale
    video_writer.py       sequential encoding
    face_detector.py      Haar cascade detection -> FaceRegion
    landmarks.py          5-point landmarks (geometric template, optional cascades)
    alignment.py          similarity transform to canonical face space
    transformer.py        source face preparation + colour matching
    blender.py            feathered / seamless compositing back onto the frame
    processor.py          pipeline orchestration, job and result types
  utils/
    config.py             dataclass config with JSON persistence
    logger.py             rotating file log + console log
  models/                 optional custom cascades (empty by default)
  assets/                 icons and sample images
  tests/                  pytest suite
  requirements.txt
```

### Pipeline

```
video_reader -> face_detector -> landmarks -> alignment -> transformer -> blender -> video_writer
                                        \____________________/
                                          processor.py drives this
```

Each stage is independently testable and takes plain numpy arrays, so a stage
can be replaced without touching the others.

## Installation (offline)

On a machine with no network access, pre-download the wheels on a connected
machine and copy them across:

```bat
pip download -r requirements.txt -d wheels --platform win_amd64 ^
    --python-version 310 --only-binary=:all:
```

Then on the target machine:

```bat
python -m pip install --no-index --find-links=wheels -r requirements.txt
```

`opencv-python` (not `opencv-python-headless`) is required on Windows so the
Haar cascade data directory is present.

## Usage

GUI:

```bat
python main.py
```

Headless conversion:

```bat
python main.py --headless --source input.mp4 --face replacement.jpg --output out.mp4
```

Environment check (useful right after an offline install):

```bat
python main.py --check
```

CLI options: `--method {copy,blend,monochrome}`, `--frame-skip N`, `--config path.json`.

## Configuration

`AppConfig` is a nested dataclass (`video`, `detection`, `processing`) saved as
JSON. Missing keys and unknown keys are tolerated, so an old config file never
breaks startup.

```python
from utils.config import AppConfig

config = AppConfig()
config.video.max_width = 640          # trade quality for speed
config.processing.frame_skip = 2      # process every other frame
config.save("config.json")
```

Useful knobs on slow machines:

| Setting | Effect |
| --- | --- |
| `video.max_width` / `max_height` | lower resolution, faster and less memory |
| `processing.frame_skip` | process every Nth frame |
| `processing.blend_strength` | 0.0 keeps the original, 1.0 full swap |
| `detection.max_faces` | cap faces handled per frame |

## Tests

```bat
python -m pytest tests -q
```

The suite runs real OpenCV operations on synthetic frames and a real video round
trip; nothing is mocked. `--check` reports whether Tkinter is available, which
is expected to be missing only on non-Windows test machines.

## Roadmap

1. Refine landmarks (eye/nose/mouth cascade refinement is wired in via
   `LandmarkExtractor(mode="cascade")`).
2. Temporal smoothing of landmarks across frames to reduce jitter
   (`FaceAligner.average_landmarks` is the hook).
3. Optional `--face-video` source so the replacement face can be animated.
4. Audio passthrough (currently video only; `ProcessingConfig.keep_audio` is a
   placeholder).
5. Optional quality mode using a compact model, kept strictly opt-in so the
   default install stays model-free and offline.
