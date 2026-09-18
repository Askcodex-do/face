"""Core pipeline for LightSwapConverter.

The package is organised as a sequence of small stages:

``video_reader`` -> ``face_detector`` -> ``landmarks`` -> ``alignment``
-> ``transformer`` -> ``blender`` -> ``video_writer``

``processor`` wires those stages together. Each stage can be imported on its
own; the names re-exported here are resolved lazily so that importing
``core.alignment`` does not drag in the video stack.
"""

from __future__ import annotations

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "AppConfig",
    "BlendResult",
    "CancelledError",
    "FaceBlender",
    "FaceBox",
    "FaceDetector",
    "FaceTransformer",
    "LandmarkEstimator",
    "Landmarks",
    "ProcessingStats",
    "ProcessorError",
    "SourceFace",
    "VideoFaceProcessor",
    "VideoReadError",
    "VideoReader",
    "VideoWriteError",
    "VideoWriter",
    "WarpedFace",
    "align_to_face",
    "read_image",
    "write_image",
]

_EXPORTS = {
    "AlignmentError": ("alignment", "AlignmentError"),
    "AlignmentResult": ("alignment", "AlignmentResult"),
    "align_to_face": ("alignment", "align_to_face"),
    "BlendResult": ("blender", "BlendResult"),
    "FaceBlender": ("blender", "FaceBlender"),
    "FaceBox": ("face_detector", "FaceBox"),
    "FaceDetector": ("face_detector", "FaceDetector"),
    "LandmarkEstimator": ("landmarks", "LandmarkEstimator"),
    "Landmarks": ("landmarks", "Landmarks"),
    "FaceTransformer": ("transformer", "FaceTransformer"),
    "WarpedFace": ("transformer", "WarpedFace"),
    "CancelledError": ("processor", "CancelledError"),
    "ProcessingStats": ("processor", "ProcessingStats"),
    "ProcessorError": ("processor", "ProcessorError"),
    "SourceFace": ("processor", "SourceFace"),
    "VideoFaceProcessor": ("processor", "VideoFaceProcessor"),
    "read_image": ("processor", "read_image"),
    "write_image": ("processor", "write_image"),
    "VideoReadError": ("video_reader", "VideoReadError"),
    "VideoReader": ("video_reader", "VideoReader"),
    "VideoWriteError": ("video_writer", "VideoWriteError"),
    "VideoWriter": ("video_writer", "VideoWriter"),
    "AppConfig": ("config", "AppConfig"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    if module_name == "config":
        from ..utils import config as module
    else:
        from importlib import import_module

        module = import_module(f".{module_name}", __name__)
    return getattr(module, attribute)
