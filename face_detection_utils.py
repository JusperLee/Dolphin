"""Utility helpers for RetinaFace-based face detection."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2

try:
    from retinaface import RetinaFace  # type: ignore
except ImportError as import_error:  # pragma: no cover - handled at runtime
    RetinaFace = None  # type: ignore
    _RETINAFACE_IMPORT_ERROR = import_error
else:
    _RETINAFACE_IMPORT_ERROR = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def _ensure_retinaface_available() -> None:
    if RetinaFace is None:  # pragma: no cover - runtime safeguard
        raise ImportError(
            "RetinaFace package is required but not installed. "
            "Install it with `pip install retinaface`."
        ) from _RETINAFACE_IMPORT_ERROR


def _to_rgb_array(image: np.ndarray, *, assume_bgr: bool = False) -> np.ndarray:
    """Convert input to an RGB numpy array."""
    if isinstance(image, np.ndarray):
        array = image
    elif Image is not None and isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"))
    else:
        raise TypeError("Expected an ndarray or PIL.Image.Image for face detection")

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Face detection expects an image with shape (H, W, 3)")

    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    if assume_bgr:
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return array


def detect_faces(
    image: np.ndarray,
    *,
    threshold: float = 0.9,
    allow_upscaling: bool = False,
    model: Optional[str] = None,
    assume_bgr: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run RetinaFace detection on an image.

    Returns bounding boxes shaped (N, 4) and confidence scores shaped (N,).
    If no face is detected, both values are ``None``.
    """
    _ensure_retinaface_available()

    rgb_image = _to_rgb_array(image, assume_bgr=assume_bgr)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    detections = RetinaFace.detect_faces(
        bgr_image,
        threshold=threshold,
        model=model,
        allow_upscaling=allow_upscaling,
    )

    if not isinstance(detections, dict) or not detections:
        return None, None

    boxes, scores = [], []
    for face_data in detections.values():
        facial_area = face_data.get("facial_area")
        if facial_area is None:
            continue
        boxes.append(facial_area)
        scores.append(face_data.get("score", 0.0))

    if not boxes:
        return None, None

    boxes_array = np.asarray(boxes, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32) if scores else None

    return boxes_array, scores_array


def extract_faces(
    image: np.ndarray,
    *,
    align: bool = True,
    threshold: float = 0.9,
    allow_upscaling: bool = False,
    model: Optional[str] = None,
    assume_bgr: bool = False,
) -> Optional[np.ndarray]:
    """Extract faces using RetinaFace.extract_faces for convenience."""
    _ensure_retinaface_available()

    rgb_image = _to_rgb_array(image, assume_bgr=assume_bgr)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    faces = RetinaFace.extract_faces(
        bgr_image,
        align=align,
        threshold=threshold,
        model=model,
        allow_upscaling=allow_upscaling,
    )

    if not faces:
        return None
    return np.asarray([np.asarray(face, dtype=np.uint8) for face in faces])
