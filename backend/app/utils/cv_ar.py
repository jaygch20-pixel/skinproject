"""
Computer Vision and Augmented Reality (AR) Processing Utilities
Combines OpenCV image processing, skin detection, lesion localization,
morphological analysis, and AR visual HUD generation.
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
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs Computer Vision skin detection and lesion localization using OpenCV.
    Calculates lesion bounding box, contour coordinates, and clinical morphology metrics.
    Generates AR visual overlay elements.
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
    
    # 2. Lesion Detection via Grayscale & Adaptive Saliency
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Otsu thresholding for pigmented lesion separation
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and distance to center
    min_area = (width * height) * 0.015
    max_area = (width * height) * 0.85
    center_x, center_y = width / 2.0, height / 2.0
    
    best_contour = None
    best_score = -1.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            # Measure proximity to center
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                proximity = 1.0 - (dist / max_dist)
                # Score combines area and central location
                score = (area / (width * height)) * 0.4 + proximity * 0.6
                if score > best_score:
                    best_score = score
                    best_contour = cnt
                    
    # Generate Bounding Box and Normalized Coordinates
    detected = False
    bbox = None
    normalized_bbox = None
    contour_points = []
    morphology = {}
    
    if best_contour is not None:
        detected = True
        bx, by, bw, bh = cv2.boundingRect(best_contour)
        bbox = {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)}
        normalized_bbox = {
            "x": round(bx / width, 4),
            "y": round(by / height, 4),
            "width": round(bw / width, 4),
            "height": round(bh / height, 4)
        }
        
        # Approximate contour polygon to smooth 20-30 points for fast AR rendering
        epsilon = 0.012 * cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, epsilon, True)
        for pt in approx:
            px, py = pt[0]
            contour_points.append({
                "x": int(px),
                "y": int(py),
                "norm_x": round(px / width, 4),
                "norm_y": round(py / height, 4)
            })
            
        # Clinical Morphology Metrics (ABCD criteria estimation aids)
        area = cv2.contourArea(best_contour)
        perimeter = cv2.arcLength(best_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        # Asymmetry score (ratio of major to minor bounding axes)
        aspect = float(bw) / float(bh) if bh > 0 else 1.0
        asymmetry_index = round(abs(1.0 - aspect), 3)
        
        morphology = {
            "candidate_area_percentage": round((area / (width * height)) * 100, 2),
            "circularity_index": round(circularity, 3),
            "bounding_box_aspect_ratio": round(aspect, 3),
            "candidate_box_max_dimension_px": int(max(bw, bh)),
            "contour_shape_heuristic": "More circular" if circularity > 0.65 else "Less circular"
        }
    else:
        morphology = {}

    lesion_mask = np.zeros((height, width), dtype=np.uint8)
    if detected and best_contour is not None:
        cv2.drawContours(lesion_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    lesion_mask = cv2.morphologyEx(
        lesion_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2
    )
    
    boundary_overlay_bgr = img_bgr.copy()
    if detected:
        cv2.drawContours(boundary_overlay_bgr, [best_contour], -1, (0, 255, 0), 3, cv2.LINE_AA)
    
    segmented_bgr = np.zeros_like(img_bgr)
    if detected:
        segmented_bgr[lesion_mask > 0] = img_bgr[lesion_mask > 0]
        
    # 3. Create AR Visual Overlay Image (Medical HUD styling)
    ar_overlay_bgr = img_bgr.copy()
    
    # AR theme colors
    hud_glow = (0, 230, 255)
    
    if detected and best_contour is not None:
        # Draw glowing contour boundary
        cv2.drawContours(ar_overlay_bgr, [best_contour], -1, hud_glow, 2, cv2.LINE_AA)
        
        # Draw AR tech brackets on bounding box
        bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        bracket_len = max(12, int(min(bw, bh) * 0.18))
        
        # Top-left
        cv2.line(ar_overlay_bgr, (bx, by), (bx + bracket_len, by), hud_glow, 3)
        cv2.line(ar_overlay_bgr, (bx, by), (bx, by + bracket_len), hud_glow, 3)
        # Top-right
        cv2.line(ar_overlay_bgr, (bx + bw, by), (bx + bw - bracket_len, by), hud_glow, 3)
        cv2.line(ar_overlay_bgr, (bx + bw, by), (bx + bw, by + bracket_len), hud_glow, 3)
        # Bottom-left
        cv2.line(ar_overlay_bgr, (bx, by + bh), (bx + bracket_len, by + bh), hud_glow, 3)
        cv2.line(ar_overlay_bgr, (bx, by + bh), (bx, by + bh - bracket_len), hud_glow, 3)
        # Bottom-right
        cv2.line(ar_overlay_bgr, (bx + bw, by + bh), (bx + bw - bracket_len, by + bh), hud_glow, 3)
        cv2.line(ar_overlay_bgr, (bx + bw, by + bh), (bx + bw, by + bh - bracket_len), hud_glow, 3)
        
        # Centroid reticle
        mcx, mcy = bx + bw // 2, by + bh // 2
        cv2.drawMarker(ar_overlay_bgr, (mcx, mcy), hud_glow, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        
    # Draw AR Status Overlay Banner
    if predicted_class and confidence is not None:
        badge_text = f"AR CV: {predicted_class.upper()} ({confidence*100:.1f}%)"
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
        border_col = (76, 175, 80) if predicted_class == "benign" else (30, 70, 245)
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
        "disclaimer": "The contour and center-zone overlay are unvalidated visual approximations, not lesion segmentation or clinical measurements. They do not constitute a diagnosis.",
        "binary_mask_data_url": binary_mask_data_url,
        "boundary_overlay_data_url": boundary_overlay_data_url,
        "segmented_image_data_url": segmented_image_data_url
    }
