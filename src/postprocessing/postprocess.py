import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import List, Tuple

def postprocess_and_draw(
    orig_image: Image.Image,
    heatmap: np.ndarray,
    box_thresh: float,
    min_area: float,
    poly_eps_ratio: float,
    unclip_ratio: float
) -> Tuple[Image.Image, List[np.ndarray]]:
    """
    Given an original PIL image and a raw haetmap, detect text regions
    and draw squares around them

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

    # 6) Draw on a copy of the original
    canvas = orig_image.copy()
    draw = ImageDraw.Draw(canvas)
    for pts in boxes:
        poly = [tuple(pt) for pt in pts]
        draw.line(poly + [poly[0]], width=5, fill="red")

    return canvas, boxes
