import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import List, Tuple

from config import TEXT_RECOGNITION_HEIGHT

def postprocess_and_draw(
    orig_image: Image.Image,
    heatmap: np.ndarray,
    box_thresh: float,
    min_area: float,
    poly_eps_ratio: float,
    unclip_ratio: float
) -> Tuple[Image.Image, List[np.ndarray], List[Image.Image]]:
    """
    Given an original PIL image and a raw haetmap, detect text regions,
    draw squares around them, and extract text crops

    Args:
        orig_image: PIL.Image resized to model input size
        heatmap: 2D array of shape (H, W) from the detector.
        box_thresh: Threshold for binarizing heatmap
        min_area: Minimum contour area to keep.
        poly_eps_ratio: Epsilon ratio for polygon approximation
        unclip_ratio: Expansion factor for each polygon

    Returns:
        canvas: PIL.Image with red polygon overlays
        boxes: List of (N*2) int arrays of polygon vertices
        crops: List of PIL.Image crops with height=TEXT_RECOGNITION_HEIGHT and preserved aspect ratio
    """
    # 1) Smooth heatmap
    smoothed = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # 2) Binarize + morphological close
    bin_map = (smoothed > box_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(bin_map, cv2.MORPH_CLOSE, kernel)

    # 3) Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    crops = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        # 4) Polygon approximation
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, poly_eps_ratio * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)

        # 5) Unclip / expand polygon
        cX, cY = pts.mean(axis=0)
        expanded = np.stack([
            [cX + (x - cX) * unclip_ratio,
             cY + (y - cY) * unclip_ratio]
            for x, y in pts
        ])
        pts_int = np.round(expanded).astype(int)
        boxes.append(pts_int)

        # Extract crop with fixed height
        x_min, y_min = pts_int.min(axis=0)
        x_max, y_max = pts_int.max(axis=0)
        crop = orig_image.crop((x_min, y_min, x_max, y_max))
        
        # Resize to fixed height while maintaining aspect ratio
        w, h = crop.size
        new_w = int(w * (TEXT_RECOGNITION_HEIGHT / h))
        crop = crop.resize((new_w, TEXT_RECOGNITION_HEIGHT), Image.Resampling.LANCZOS)
        crops.append(crop)

    # 6) Draw on a copy of the original
    canvas = orig_image.copy()
    draw = ImageDraw.Draw(canvas)
    for pts in boxes:
        poly = [tuple(pt) for pt in pts]
        draw.line(poly + [poly[0]], width=5, fill="red")

    return canvas, boxes, crops
