import io
from PIL import Image

from config import (
    MODEL_PATH,
    BOX_THRESH,
    MIN_AREA,
    POLY_EPS_RATIO,
    UNCLIP_RATIO,
)
from loader.model_loader import load_session
from preprocessing.preprocess import preprocess_image
from detection.detector import run_detection
from postprocessing.postprocess import postprocess_and_draw
from ui.components import (
    render_header,
    file_uploader,
    show_heatmap_info,
    show_detected_image,
)

def main():
    # 1) Header
    render_header()

    # 2) Load model (cached)
    session = load_session(str(MODEL_PATH))

    # 3) Upload & read
    img_bytes = file_uploader()
    if img_bytes is None:
        return

    # 4) Preprocess
    input_tensor = preprocess_image(img_bytes)

    # 5) Inference
    heatmap = run_detection(session, input_tensor)
    show_heatmap_info(heatmap)

    # 6) Prepare original image for drawing
    orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig = orig.resize((input_tensor.shape[-1], input_tensor.shape[-2]), Image.BILINEAR)

    # 7) Postprocess & draw
    vis_img, boxes = postprocess_and_draw(
        orig_image=orig,
        heatmap=heatmap,
        box_thresh=BOX_THRESH,
        min_area=MIN_AREA,
        poly_eps_ratio=POLY_EPS_RATIO,
        unclip_ratio=UNCLIP_RATIO,
    )

    # 8) Display results
    show_detected_image(vis_img, caption="Detected text regions", boxes=boxes)


if __name__ == "__main__":
    main()
