from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_PATH   = MODEL_DIR / "ml_PP-OCRv3_det.onnx"

# === Image preprocessing ===
IMAGE_SIZE   = (960, 960)    # width, height
MEAN         = (0.485, 0.456, 0.406)
STD          = (0.229, 0.224, 0.225)

# === Detection thresholds ===
BOX_THRESH       = 0.2
MIN_AREA         = 300
POLY_EPS_RATIO   = 0.01
UNCLIP_RATIO     = 1.8

# === Text recognition settings ===
TEXT_RECOGNITION_HEIGHT = 48  # Required height for text recognition

# === Streamlit UI settings ===
STREAMLIT_TITLE  = "Text Detection Demo"
UPLOAD_TYPES     = ["jpg", "png", "jpeg"]
