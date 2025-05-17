import streamlit as st
from PIL import Image
from typing import List, Union
import numpy as np

from config import STREAMLIT_TITLE, UPLOAD_TYPES

def render_header():
    """Render the page title"""
    st.title(STREAMLIT_TITLE)

def file_uploader() -> Union[bytes, None]:
    """
    Show a file uploader and return the raw bytes of the uploaded image (once).
    Returns None if no file is uploaded
    """
    upload = st.file_uploader(
        "Upload an image", 
        type=UPLOAD_TYPES,
        accept_multiple_files=False
    )
    if upload is not None:
        return upload.read()
    return None

def show_heatmap_info(heatmap: np.ndarray):
    """show infomration about the heatmap"""
    st.write(
        f"Heatmap shape: {heatmap.shape}, "
        f"min/max = {heatmap.min():.3f}/{heatmap.max():.3f}"
    )

def show_detected_image(
    image: Image.Image, 
    caption: str, 
    boxes: List[np.ndarray]
):
    """
    Display the image with detected boxes and a count
    """
    st.image(image, caption=caption, use_container_width=True)
    st.write(f"Found {len(boxes)} text boxes")
