from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np


@dataclass
class FrameMeta:
    """Metadata for a single video frame."""
    index: int
    timestamp_ms: float
    width: int
    height: int


@dataclass
class Frame:
    """A video frame with its pixel data and metadata."""
    image: np.ndarray        # HWC uint8 BGR (OpenCV convention)
    meta: FrameMeta


class VideoReader:
    """OpenCV-based video reader. Supports files, RTSP streams, webcams.

    Usage::

        with VideoReader("input.mp4") as reader:
            for frame in reader:
                process(frame.image)
    """

    def __init__(
        self,
        source: str | Path | int,
        *,
        sample_fps: float | None = None,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
    ) -> None:
        self._source = str(source) if not isinstance(source, int) else source
        self._sample_fps = sample_fps
        self._max_frames = max_frames
        self._resize = resize  # (width, height)
        self._cap: Any = None

    # ── Properties ────────────────────────────────────────

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        import cv2
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def frame_count(self) -> int:
        if self._cap is None:
            return 0
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        if self._cap is None:
            return 0
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            return 0
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_seconds(self) -> float:
        fps = self.fps
        if fps <= 0:
            return 0.0
        return self.frame_count / fps

    # ── Context manager ───────────────────────────────────

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self._source}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Iteration ─────────────────────────────────────────

    def __iter__(self) -> Iterator[Frame]:
        import cv2

        if self._cap is None:
            self.open()
        assert self._cap is not None

        native_fps = self.fps
        sample_interval = 1
        if self._sample_fps is not None and self._sample_fps > 0 and native_fps > 0:
            sample_interval = max(1, int(round(native_fps / self._sample_fps)))

        frame_idx = 0
        yielded = 0

        while True:
            if self._max_frames is not None and yielded >= self._max_frames:
                break

            ret, img = self._cap.read()
            if not ret or img is None:
                break

            if frame_idx % sample_interval == 0:
                if self._resize is not None:
                    img = cv2.resize(img, self._resize, interpolation=cv2.INTER_LINEAR)

                timestamp_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
                h, w = img.shape[:2]
                meta = FrameMeta(index=frame_idx, timestamp_ms=timestamp_ms, width=w, height=h)
                yield Frame(image=img, meta=meta)
                yielded += 1

            frame_idx += 1

    # ── Utilities ─────────────────────────────────────────

    def read_frame(self, index: int) -> Frame | None:
        """Read a specific frame by index (seek)."""
        import cv2

        if self._cap is None:
            self.open()
        assert self._cap is not None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, img = self._cap.read()
        if not ret or img is None:
            return None

        if self._resize is not None:
            img = cv2.resize(img, self._resize, interpolation=cv2.INTER_LINEAR)

        timestamp_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w = img.shape[:2]
        return Frame(
            image=img,
            meta=FrameMeta(index=index, timestamp_ms=timestamp_ms, width=w, height=h),
        )


class VideoWriter:
    """OpenCV-based video writer. Outputs MP4 (H.264) by default.

    Usage::

        with VideoWriter("output.mp4", fps=30, size=(1920, 1080)) as writer:
            writer.write(frame_bgr)
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        size: tuple[int, int] | None = None,  # (width, height)
        codec: str = "mp4v",
    ) -> None:
        self._path = Path(output_path)
        self._fps = fps
        self._size = size
        self._codec = codec
        self._writer: Any = None
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """Write a BGR frame. Auto-initializes on first call if size not set."""
        import cv2

        if self._writer is None:
            h, w = frame.shape[:2]
            if self._size is None:
                self._size = (w, h)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self._codec)
            self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, self._size)
            if not self._writer.isOpened():
                raise IOError(f"Cannot open video writer: {self._path}")

        self._writer.write(frame)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def extract_frames(
    source: str | Path | int,
    output_dir: str | Path,
    *,
    sample_fps: float | None = None,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
    format: str = "jpg",
    quality: int = 95,
) -> dict[str, Any]:
    """Extract frames from video to image files.

    Returns summary dict with frame count, output dir, etc.
    """
    import cv2

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    with VideoReader(source, sample_fps=sample_fps, max_frames=max_frames, resize=resize) as reader:
        total = reader.frame_count
        native_fps = reader.fps

        for frame in reader:
            filename = f"frame_{frame.meta.index:08d}.{format}"
            path = out / filename
            if format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(path), frame.image)
            saved += 1

    return {
        "source": str(source),
        "output_dir": str(out),
        "frames_saved": saved,
        "native_fps": native_fps,
        "sample_fps": sample_fps,
        "total_source_frames": total,
    }
