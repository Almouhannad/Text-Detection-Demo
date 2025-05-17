import io
import numpy as np
from PIL import Image
from src.config import IMAGE_SIZE, MEAN, STD

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a normalized CHW (Channel Height Width) tensor with batch dimension

    Args:
        image_bytes: Raw JPEG/PNG bytes
    Returns:
        np.ndarray shaped (1, 3, H, W), dtype float32
    """
    # 1. Load & convert
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Resize
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)

    # 3. To array & scale
    im_arr = np.array(img, dtype=np.float32) / 255.0

    # 4. Normalize
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    im_arr = (im_arr - mean) / std

    # 5. HWC → CHW, add batch dimension
    tensor = np.transpose(im_arr, (2, 0, 1))[None, ...]

    return tensor
