"""Core processing layer.

Modules are intentionally decoupled: the GUI never touches OpenCV directly and
every stage can be unit tested with plain numpy arrays.
"""

from .alignment import AlignmentResult, FaceAligner
from .blender import FaceBlender
from .face_detector import FaceDetector, FaceRegion
from .landmarks import FaceLandmarks, LandmarkExtractor
from .processor import ConversionJob, ConversionResult, Processor
from .transformer import FaceTransformer, TransformResult
from .video_reader import FrameInfo, VideoInfo, VideoReader
from .video_writer import VideoWriter

__all__ = [
    "VideoReader",
    "VideoInfo",
    "FrameInfo",
    "VideoWriter",
    "FaceDetector",
    "FaceRegion",
    "LandmarkExtractor",
    "FaceLandmarks",
    "FaceAligner",
    "AlignmentResult",
    "FaceTransformer",
    "TransformResult",
    "FaceBlender",
    "Processor",
    "ConversionJob",
    "ConversionResult",
]
