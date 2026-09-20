"""Frame pipeline orchestration.

Ties the stages together: detect -> landmarks -> align -> transform -> blend.
The processor is deliberately synchronous and single threaded. On a 2 GB CPU
only machine a background worker would add little throughput while making
cancellation and progress reporting harder to get right; the GUI simply calls
``process`` from a worker thread and consumes the progress callback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from core.alignment import FaceAligner
from core.blender import FaceBlender
from core.face_detector import FaceDetector, FaceRegion
from core.landmarks import FaceLandmarks, LandmarkExtractor
from core.transformer import FaceTransformer, TransformResult, create_transformer
from core.video_reader import VideoInfo, VideoReader
from core.video_writer import VideoWriter
from utils.config import AppConfig
from utils.logger import get_logger

logger = get_logger("core.processor")

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class ConversionJob:
    """Everything needed to run one conversion."""

    source_video: str
    target_face_image: str
    output_video: Optional[str] = None
    source_landmarks: Optional[FaceLandmarks] = None
    frame_skip: int = 1
    method: str = "copy"
    strength: float = 0.85
    color_match: bool = True
    codec: str = "mp4v"
    max_width: int = 960
    max_height: int = 540


@dataclass
class ConversionResult:
    """Summary returned after a conversion finishes."""

    output_path: str
    frames_read: int = 0
    frames_written: int = 0
    faces_detected: int = 0
    cancelled: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return not self.cancelled and not self.errors and self.frames_written > 0


class Processor:
    """Run the face replacement pipeline over a whole video."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()

        detection = self.config.detection
        processing = self.config.processing
        video = self.config.video

        self.detector = FaceDetector(
            scale_factor=detection.scale_factor,
            min_neighbors=detection.min_neighbors,
            min_size=detection.min_size,
            max_faces=detection.max_faces,
        )
        self.landmarks = LandmarkExtractor(mode="template", detector=self.detector)
        self.aligner = FaceAligner()
        self.blender = FaceBlender(
            strength=processing.blend_strength,
            feather=processing.mask_feather,
        )
        self.default_frame_skip = max(int(processing.frame_skip), 1)
        self.default_max_size = (video.max_width, video.max_height)
        self.default_codec = video.codec

    # --------------------------------------------------------------- pipeline
    def process_frame(
        self,
        frame: np.ndarray,
        prepared_source: TransformResult,
        transformer: FaceTransformer,
    ) -> tuple[np.ndarray, int]:
        """Run the full pipeline on one frame.

        Returns the modified frame and the number of faces replaced.
        """
        faces = self.detector.detect(frame)
        if not faces:
            return frame, 0

        replaced = 0
        for face in faces:
            face_landmarks = self.landmarks.extract(frame, face)
            if not face_landmarks.is_valid:
                continue

            alignment = self.aligner.compute_matrix(face_landmarks)
            if not alignment.valid:
                continue

            patch, _ = self.aligner.align(frame, face_landmarks)
            transformed = transformer.transform(prepared_source, patch, face_landmarks)
            result = self.blender.blend(frame, transformed.image, transformed.mask, alignment)
            if result.applied:
                replaced += 1

        return frame, replaced

    def prepare_source_face(self, image_path: str, method: str = "copy") -> TransformResult:
        """Detect and align the replacement face from a still image."""
        transformer = create_transformer(
            method,
            color_match=self.config.processing.color_match,
            aligner=self.aligner,
        )
        source_image = transformer.load_source(image_path)
        face = self.detector.detect_primary(source_image)
        if face is None:
            logger.warning("No face found in source image %s; using the full image.", image_path)
            return transformer.prepare(source_image, None)  # type: ignore[arg-type]

        source_landmarks = self.landmarks.extract(source_image, face)
        return transformer.prepare(source_image, source_landmarks)

    # ------------------------------------------------------------- conversion
    def process(
        self,
        job: ConversionJob,
        progress: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCallback] = None,
    ) -> ConversionResult:
        """Convert a whole video, streaming frames to the output file."""
        output_path = job.output_video or self.default_output_path(job.source_video)
        result = ConversionResult(output_path=str(output_path))

        transformer = create_transformer(
            job.method,
            color_match=job.color_match,
            aligner=self.aligner,
        )

        frame_skip = max(int(job.frame_skip or self.default_frame_skip), 1)
        max_size = (job.max_width or self.default_max_size[0], job.max_height or self.default_max_size[1])

        reader = VideoReader(
            job.source_video,
            max_width=max_size[0],
            max_height=max_size[1],
            fps_fallback=self.config.video.fps_fallback,
        )

        writer = None
        info: Optional[VideoInfo] = None
        try:
            info = reader.open()
            writer = VideoWriter(
                output_path,
                fps=info.fps / frame_skip,
                size=(info.width, info.height),
                codec=job.codec or self.default_codec,
            )
            prepared = self.prepare_source_face(job.target_face_image, job.method)
            with writer:
                for frame, frame_info in reader.frames():
                    if is_cancelled is not None and is_cancelled():
                        result.cancelled = True
                        logger.info("Conversion cancelled by caller.")
                        break

                    result.frames_read += 1
                    if frame_info.index % frame_skip != 0:
                        continue

                    processed, replaced = self.process_frame(frame, prepared, transformer)
                    result.faces_detected += replaced
                    if writer.write(processed):
                        result.frames_written += 1

                    if progress is not None:
                        progress(frame_info.index, info.frame_count, "Processing frames")
        except Exception as exc:  # surfaced to the GUI rather than crashing it
            logger.exception("Conversion failed: %s", exc)
            result.errors.append(str(exc))
        finally:
            reader.close()

        if progress is not None and info is not None:
            progress(info.frame_count, info.frame_count, "Done")
        return result

    def default_output_path(self, source_video: str) -> Path:
        """Pick a non-destructive default output name in the output folder."""
        source = Path(source_video)
        output_dir = self.config.output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{source.stem}_swapped.{self.config.video.container}"

    # -------------------------------------------------------------- utilities
    def estimate_frames(self, video_path: str) -> int:
        """Cheap frame count probe used for progress bars before a job starts."""
        reader = VideoReader(video_path)
        try:
            return reader.open().frame_count
        except Exception:
            return 0
        finally:
            reader.close()

    @staticmethod
    def draw_faces(frame: np.ndarray, faces: List[FaceRegion]) -> np.ndarray:
        """Debug helper that outlines detected faces in place."""
        import cv2

        for face in faces:
            cv2.rectangle(frame, (face.x, face.y), (face.x2, face.y2), (0, 255, 0), 2)
        return frame
