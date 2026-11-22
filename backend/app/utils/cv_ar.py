"""
Computer Vision and Augmented Reality (AR) Processing Utilities
Combines OpenCV image processing, approximate skin-context candidates,
and AR visual output generation.
"""

import io
import base64
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image

def pil_to_jpeg_data_url(pil_image, quality=92):
    rgb = pil_image.convert("RGB")
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def pil_to_png_data_url(pil_image):
    rgb = pil_image.convert("RGB")
    _, buffer = cv2.imencode(".png", cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def analyze_skin_and_lesion(
    pil_image: Image.Image,
    predicted_class: Optional[str] = None,
    confidence: Optional[float] = None,
    generate_segmentation: bool = True
) -> Dict[str, Any]:
    """
    Finds conservative intensity candidates within estimated skin context.
    These candidates are not model-generated or validated disease boundaries.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    height, width, _ = img_rgb.shape
    
    # Convert to OpenCV BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Skin Detection in YCrCb color space
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 130, 75], dtype=np.uint8)
    upper_skin = np.array([255, 175, 130], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    
    skin_pixel_count = cv2.countNonZero(skin_mask)
    skin_ratio = skin_pixel_count / (width * height)
    has_skin = skin_ratio > 0.08
    
    # 2. Generate conservative intensity candidates, constrained to skin context.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skin_context = cv2.dilate(
        skin_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1
    )
    candidate_mask = cv2.bitwise_and(otsu_thresh, skin_context)
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )

    raw_contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = width * height
    min_area = max(24, image_area * 0.002)
    selected_contours = []
    morphology = {}

    if generate_segmentation and has_skin:
        for contour in raw_contours:
            area = cv2.contourArea(contour)
            bx, by, bw, bh = cv2.boundingRect(contour)
            if area < min_area or area > image_area * 0.35:
                continue
            if bx <= 1 or by <= 1 or bx + bw >= width - 1 or by + bh >= height - 1:
                continue
            if min(bw, bh) == 0 or max(bw / bh, bh / bw) > 5:
                continue
            if area / max(bw * bh, 1) < 0.12:
                continue

            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, cv2.FILLED)
            surrounding = cv2.dilate(
                component_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                iterations=1
            )
            ring = (surrounding > 0) & (component_mask == 0) & (skin_context > 0)
            inside = (component_mask > 0) & (skin_context > 0)
            if np.count_nonzero(ring) < min_area or np.count_nonzero(inside) < min_area:
                continue
            # Reject low-contrast structures; thin hairs and threshold noise are common false edges.
            if float(np.mean(gray[ring]) - np.mean(gray[inside])) < 8:
                continue
            selected_contours.append(contour)

    detected = bool(selected_contours)
    bbox = None
    normalized_bbox = None
    contour_points = []
    contour_polygons = []
    if detected:
        all_points = np.concatenate(selected_contours, axis=0)
        bx, by, bw, bh = cv2.boundingRect(all_points)
        bbox = {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)}
        normalized_bbox = {
            "x": round(bx / width, 4),
            "y": round(by / height, 4),
            "width": round(bw / width, 4),
            "height": round(bh / height, 4)
        }
        for contour in selected_contours:
            epsilon = max(1.0, 0.012 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [
                {
                    "x": int(point[0][0]),
                    "y": int(point[0][1]),
                    "norm_x": round(float(point[0][0]) / width, 6),
                    "norm_y": round(float(point[0][1]) / height, 6)
                }
                for point in approx
            ]
            if len(points) >= 3:
                contour_polygons.append(points)
        contour_points = contour_polygons[0] if contour_polygons else []
        morphology = {
            "candidate_regions": len(contour_polygons),
            "candidate_area_percentage": round(
                sum(cv2.contourArea(contour) for contour in selected_contours)
                / image_area * 100,
                2
            )
        }

    lesion_mask = np.zeros((height, width), dtype=np.uint8)
    if detected:
        cv2.drawContours(lesion_mask, selected_contours, -1, 255, thickness=cv2.FILLED)
    
    boundary_overlay_bgr = img_bgr.copy()
    if detected:
        cv2.drawContours(boundary_overlay_bgr, selected_contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    
    segmented_bgr = np.zeros_like(img_bgr)
    if detected:
        segmented_bgr[lesion_mask > 0] = img_bgr[lesion_mask > 0]
        
    # 3. Create AR Visual Overlay Image (Medical HUD styling)
    ar_overlay_bgr = img_bgr.copy()
    
    # AR theme colors
    hud_glow = (0, 230, 255)
    
    if detected:
        # Draw glowing contour boundary
        cv2.drawContours(ar_overlay_bgr, selected_contours, -1, hud_glow, 2, cv2.LINE_AA)
        
    # Draw AR Status Overlay Banner
    if predicted_class and confidence is not None:
        badge_text = f"MODEL: {predicted_class.upper()} ({confidence*100:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(width, height) / 550.0)
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
        
        # Semi-transparent badge background
        overlay = ar_overlay_bgr.copy()
        bx1, by1 = 12, 12
        bx2, by2 = bx1 + tw + 16, by1 + th + 18
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, ar_overlay_bgr, 0.25, 0, ar_overlay_bgr)
        
        # Border & Text
        border_col = (76, 175, 80) if predicted_class == "benign" else (
            (120, 120, 120) if predicted_class == "uncertain" else (30, 70, 245)
        )
        cv2.rectangle(ar_overlay_bgr, (bx1, by1), (bx2, by2), border_col, 1)
        cv2.putText(ar_overlay_bgr, badge_text, (bx1 + 8, by1 + th + 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    # Convert overlay to base64 JPEG
    _, buffer = cv2.imencode(".jpg", ar_overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")
    overlay_data_url = f"data:image/jpeg;base64,{overlay_base64}"

    _, mask_buffer = cv2.imencode(".png", lesion_mask)
    binary_mask_data_url = "data:image/png;base64," + base64.b64encode(mask_buffer).decode("utf-8")

    _, boundary_buffer = cv2.imencode(".jpg", boundary_overlay_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 92])
    boundary_overlay_data_url = "data:image/jpeg;base64," + base64.b64encode(boundary_buffer).decode("utf-8")

    _, segmented_buffer = cv2.imencode(".png", segmented_bgr)
    segmented_image_data_url = "data:image/png;base64," + base64.b64encode(segmented_buffer).decode("utf-8")
    
    return {
        "detected": detected,
        "has_skin_pigment": has_skin,
        "skin_coverage_ratio": round(skin_ratio, 3),
        "bounding_box": bbox,
        "normalized_bounding_box": normalized_bbox,
        "contour_points": contour_points,
        "morphology": morphology,
        "ar_overlay_data_url": overlay_data_url,
        "contours": contour_polygons,
        "localization_supported": False,
        "localization_method": "Conservative OpenCV intensity candidate within estimated skin context; not disease localization.",
        "disclaimer": "This OpenCV intensity candidate is approximate and is not a disease segmentation, validated lesion boundary, or clinical measurement.",
        "binary_mask_data_url": binary_mask_data_url,
        "boundary_overlay_data_url": boundary_overlay_data_url,
        "segmented_image_data_url": segmented_image_data_url
    }
