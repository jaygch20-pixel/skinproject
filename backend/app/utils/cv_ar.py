"""
Computer vision utilities for approximate lesion candidate visualization.

The classifier in this project predicts image-level categories only. The
OpenCV mask produced here is a visual candidate, not a validated lesion mask.
"""

import base64
import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

MAX_ANALYSIS_SIDE = 640
AR_CONTOUR_POINTS = 48
logger = logging.getLogger(__name__)
SEGMENTATION_WARNING = (
    "Lesion boundary could not be confidently detected. "
    "Try a clearer image with better lighting."
)


def _encode_data_url(image: np.ndarray, extension: str, mime_type: str) -> str:
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Could not encode generated {mime_type} image.")
    return f"data:{mime_type};base64," + base64.b64encode(buffer).decode("ascii")


def pil_to_jpeg_data_url(pil_image: Image.Image, quality: int = 92) -> str:
    rgb = np.asarray(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise ValueError("Could not encode JPEG image.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def pil_to_png_data_url(pil_image: Image.Image) -> str:
    rgb = np.asarray(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return _encode_data_url(bgr, ".png", "image/png")


def _resample_contour(contour: np.ndarray, point_count: int) -> np.ndarray:
    points = contour[:, 0, :].astype(np.float32)
    if len(points) < 3:
        return points
    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segment_lengths.sum())
    if perimeter <= 0:
        return points[:point_count]
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distances = np.linspace(0.0, perimeter, point_count, endpoint=False)
    sampled = np.empty((point_count, 2), dtype=np.float32)
    for axis in range(2):
        sampled[:, axis] = np.interp(
            distances, cumulative, closed[:, axis]
        )
    return sampled


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    chroma_candidates = cv2.inRange(
        ycrcb,
        np.array([0, 115, 60], dtype=np.uint8),
        np.array([255, 190, 150], dtype=np.uint8)
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    skin_hue = ((hue <= 30) | (hue >= 170)) & (hsv[:, :, 1] >= 12)
    return cv2.bitwise_and(chroma_candidates, skin_hue.astype(np.uint8) * 255)


def _candidate_mask(
    rgb: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
    """Return one central, contrast-supported region candidate and its score."""
    height, width = rgb.shape[:2]
    image_area = height * width
    short_side = min(height, width)
    skin_mask = _skin_mask(rgb)
    skin_ratio = float(cv2.countNonZero(skin_mask)) / image_area
    if skin_ratio < 0.08:
        return np.zeros((height, width), dtype=np.uint8), None, 0.0, skin_ratio

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if short_side >= 5:
        median_size = 5 if short_side >= 32 else 3
        lab = cv2.medianBlur(lab, median_size)

    sigma = max(3.0, short_side * 0.10)
    background = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    difference = cv2.absdiff(lab, background).astype(np.float32)
    saliency = cv2.magnitude(
        difference[:, :, 0] * 0.55,
        cv2.magnitude(difference[:, :, 1], difference[:, :, 2])
    )
    saliency_u8 = np.clip(saliency, 0, 255).astype(np.uint8)

    roi_margin_x = max(1, int(width * 0.04))
    roi_margin_y = max(1, int(height * 0.04))
    roi = saliency_u8[
        roi_margin_y:height - roi_margin_y,
        roi_margin_x:width - roi_margin_x
    ]
    if roi.size == 0:
        return np.zeros((height, width), dtype=np.uint8), None, 0.0, skin_ratio

    otsu_threshold, _ = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    median = float(np.median(roi))
    mad = float(np.median(np.abs(roi.astype(np.float32) - median)))
    robust_threshold = median + max(3.0, 2.5 * 1.4826 * mad)
    high_contrast_floor = float(np.percentile(roi, 99.5)) * 0.65
    threshold = max(
        6.0, float(otsu_threshold), robust_threshold, high_contrast_floor
    )
    if threshold >= 80:
        return np.zeros((height, width), dtype=np.uint8), None, 0.0, skin_ratio

    candidate = cv2.threshold(
        saliency_u8, threshold, 255, cv2.THRESH_BINARY
    )[1]
    kernel_size = min(11, max(3, int(round(short_side * 0.025)) | 1))
    open_size = max(3, min(5, (short_side // 100) * 2 + 1))
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )

    component_count, labels, stats, centers = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    min_area = max(16, int(image_area * 0.002))
    max_area = int(image_area * 0.60)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    center_scale = max(1.0, np.hypot(width, height) * 0.45)
    best_label = None
    best_score = 0.0
    best_confidence = 0.0

    for label in range(1, component_count):
        x, y, box_width, box_height, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if box_width <= 0 or box_height <= 0:
            continue
        if x <= 0 or y <= 0 or x + box_width >= width - 1 or y + box_height >= height - 1:
            continue
        aspect_ratio = max(box_width / box_height, box_height / box_width)
        fill_ratio = float(area) / max(1, box_width * box_height)
        if aspect_ratio > 5.0 or fill_ratio < 0.18:
            continue

        cx, cy = centers[label]
        center_distance = float(np.linalg.norm(np.array([cx, cy]) - image_center))
        centrality = float(np.exp(-center_distance / center_scale))
        component = labels == label
        context_margin = max(8, int(max(box_width, box_height) * 0.20))
        context_x0 = max(0, x - context_margin)
        context_y0 = max(0, y - context_margin)
        context_x1 = min(width, x + box_width + context_margin)
        context_y1 = min(height, y + box_height + context_margin)
        context_component = component[context_y0:context_y1, context_x0:context_x1]
        context_skin = skin_mask[context_y0:context_y1, context_x0:context_x1] > 0
        surrounding = ~context_component
        if not np.any(surrounding):
            continue
        surrounding_skin_ratio = float(np.count_nonzero(context_skin & surrounding)) / np.count_nonzero(surrounding)
        if surrounding_skin_ratio < 0.20:
            continue
        component_saliency = float(np.mean(saliency[component]))
        contrast_score = min(1.0, component_saliency / max(18.0, threshold * 2.0))
        size_score = min(1.0, float(area) / max(min_area * 8, 1))
        score = (
            0.40 * contrast_score +
            0.25 * centrality +
            0.15 * min(1.0, fill_ratio) +
            0.20 * size_score
        )
        if score > best_score:
            best_score = score
            best_label = label
            best_confidence = score

    if best_label is None or best_confidence < 0.30:
        return np.zeros((height, width), dtype=np.uint8), None, best_confidence, skin_ratio

    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros((height, width), dtype=np.uint8), None, 0.0, skin_ratio
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    x, y, box_width, box_height = cv2.boundingRect(contour)
    margin = max(6, int(max(box_width, box_height) * 0.12))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + box_width + margin)
    y1 = min(height, y + box_height + margin)
    roi_candidate = mask[y0:y1, x0:x1]
    roi_rgb = rgb[y0:y1, x0:x1]
    grabcut_kernel_size = min(31, max(3, int(max(box_width, box_height) * 0.05) | 1))
    grabcut_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (grabcut_kernel_size, grabcut_kernel_size)
    )
    probable_foreground = roi_candidate > 0
    definite_foreground = cv2.erode(
        roi_candidate, grabcut_kernel, iterations=1
    ) > 0
    if not np.any(definite_foreground):
        distance = cv2.distanceTransform(roi_candidate, cv2.DIST_L2, 3)
        definite_foreground = distance >= max(1.0, float(distance.max()) * 0.35)
    dilated_candidate = cv2.dilate(
        roi_candidate, grabcut_kernel, iterations=1
    ) > 0
    grabcut_mask = np.full(
        roi_candidate.shape, cv2.GC_PR_BGD, dtype=np.uint8
    )
    grabcut_mask[probable_foreground] = cv2.GC_PR_FGD
    grabcut_mask[definite_foreground] = cv2.GC_FGD
    grabcut_mask[~dilated_candidate] = cv2.GC_BGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_MASK
        )
        refined = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)
        refined_contours, _ = cv2.findContours(
            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if refined_contours:
            refined_contour = max(refined_contours, key=cv2.contourArea)
            refined_area = cv2.contourArea(refined_contour)
            original_area = cv2.contourArea(contour)
            if min_area <= refined_area <= max_area and refined_area <= original_area * 1.15:
                mask.fill(0)
                refined_contour[:, 0, 0] += x0
                refined_contour[:, 0, 1] += y0
                cv2.drawContours(mask, [refined_contour], -1, 255, cv2.FILLED)
                contour = refined_contour
    except cv2.error as error:
        logger.warning("GrabCut refinement failed; retaining threshold candidate: %s", error)

    hole_fill = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(hole_fill, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(hole_fill)
    mask = cv2.bitwise_or(mask, holes)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea) if contours else None
    if contour is None or cv2.contourArea(contour) < min_area:
        return np.zeros((height, width), dtype=np.uint8), None, 0.0, skin_ratio
    return mask, contour, best_confidence, skin_ratio


def analyze_skin_and_lesion(
    pil_image: Image.Image,
    predicted_class: Optional[str] = None,
    confidence: Optional[float] = None,
    include_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Create a conservative OpenCV color/intensity candidate and aligned outputs.
    No mask is returned as a successful visualization when confidence is low.
    """
    source_rgb = np.asarray(pil_image.convert("RGB"))
    source_height, source_width = source_rgb.shape[:2]
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(source_width, source_height))
    if scale < 1.0:
        analysis_width = max(1, round(source_width * scale))
        analysis_height = max(1, round(source_height * scale))
        analysis_rgb = cv2.resize(
            source_rgb, (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        analysis_rgb = source_rgb.copy()

    small_mask, small_contour, candidate_score, skin_ratio = _candidate_mask(analysis_rgb)
    detected = small_contour is not None
    if detected:
        mask = cv2.resize(
            small_mask, (source_width, source_height),
            interpolation=cv2.INTER_NEAREST
        )
        scale_x = source_width / analysis_rgb.shape[1]
        scale_y = source_height / analysis_rgb.shape[0]
        contour = small_contour.astype(np.float32)
        contour[:, 0, 0] *= scale_x
        contour[:, 0, 1] *= scale_y
        contour = np.rint(contour).astype(np.int32)
        contour[:, 0, 0] = np.clip(contour[:, 0, 0], 0, source_width - 1)
        contour[:, 0, 1] = np.clip(contour[:, 0, 1], 0, source_height - 1)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        bx, by, box_width, box_height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center = {
                "x": round(moments["m10"] / moments["m00"], 2),
                "y": round(moments["m01"] / moments["m00"], 2)
            }
        else:
            center = {"x": round(bx + box_width / 2, 2), "y": round(by + box_height / 2, 2)}
        sampled = _resample_contour(contour, AR_CONTOUR_POINTS)
        contour_points = [
            {
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "norm_x": round(float(point[0]) / source_width, 6),
                "norm_y": round(float(point[1]) / source_height, 6)
            }
            for point in sampled
        ]
        normalized_bbox = {
            "x": round(bx / source_width, 6),
            "y": round(by / source_height, 6),
            "width": round(box_width / source_width, 6),
            "height": round(box_height / source_height, 6)
        }
        bbox = {
            "x": int(bx), "y": int(by),
            "width": int(box_width), "height": int(box_height)
        }
        contour_area = float(cv2.contourArea(contour))
        morphology = {
            "candidate_area_percentage": round(
                contour_area / (source_width * source_height) * 100, 2
            ),
            "candidate_regions": 1,
            "candidate_score": round(float(candidate_score), 3)
        }
    else:
        mask = np.zeros((source_height, source_width), dtype=np.uint8)
        contour = None
        contour_points = []
        normalized_bbox = None
        bbox = None
        center = None
        morphology = {}

    overlay_rgb = source_rgb.copy()
    boundary_rgb = source_rgb.copy()
    if detected:
        blue_bgr = (255, 0, 0)
        source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
        tint = source_bgr.copy()
        tint[mask > 0] = (255, 90, 30)
        blended = cv2.addWeighted(tint, 0.27, source_bgr, 0.73, 0)
        cv2.drawContours(blended, [contour], -1, blue_bgr, 2, cv2.LINE_AA)
        cv2.rectangle(
            blended, (bbox["x"], bbox["y"]),
            (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
            (255, 180, 0), 1, cv2.LINE_AA
        )
        cv2.circle(
            blended, (int(round(center["x"])), int(round(center["y"]))),
            max(3, round(min(source_width, source_height) * 0.008)),
            (255, 255, 255), -1, cv2.LINE_AA
        )
        overlay_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        boundary_bgr = source_bgr.copy()
        cv2.drawContours(boundary_bgr, [contour], -1, blue_bgr, 3, cv2.LINE_AA)
        cv2.rectangle(
            boundary_bgr, (bbox["x"], bbox["y"]),
            (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
            (255, 180, 0), 1, cv2.LINE_AA
        )
        cv2.circle(
            boundary_bgr, (int(round(center["x"])), int(round(center["y"]))),
            max(3, round(min(source_width, source_height) * 0.008)),
            (255, 255, 255), -1, cv2.LINE_AA
        )
        boundary_rgb = cv2.cvtColor(boundary_bgr, cv2.COLOR_BGR2RGB)

    result: Dict[str, Any] = {
        "detected": detected,
        "has_skin_pigment": skin_ratio > 0.08,
        "skin_coverage_ratio": round(skin_ratio, 3),
        "image_width": source_width,
        "image_height": source_height,
        "bounding_box": bbox,
        "normalized_bounding_box": normalized_bbox,
        "center": center,
        "contour_points": contour_points,
        "contours": [contour_points] if detected else [],
        "morphology": morphology,
        "segmentation_warning": None if detected else SEGMENTATION_WARNING,
        "localization_supported": False,
        "localization_method": (
            "OpenCV Lab local-contrast candidate; not model-generated segmentation."
        ),
        "disclaimer": (
            "OpenCV candidate only. It is not a validated disease segmentation "
            "or clinical boundary."
        )
    }

    if include_visualizations:
        result.update({
            "ar_overlay_data_url": pil_to_jpeg_data_url(Image.fromarray(overlay_rgb)),
            "binary_mask_data_url": (
                _encode_data_url(mask, ".png", "image/png") if detected else None
            ),
            "boundary_overlay_data_url": (
                pil_to_jpeg_data_url(Image.fromarray(boundary_rgb)) if detected else None
            ),
            "segmented_image_data_url": (
                pil_to_png_data_url(Image.fromarray(overlay_rgb)) if detected else None
            )
        })
    return result
