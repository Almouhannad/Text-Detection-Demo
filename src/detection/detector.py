import numpy as np
from onnxruntime import InferenceSession

def run_detection(
    session: InferenceSession, 
    input_tensor: np.ndarray
) -> np.ndarray:
    """
    Run the OCR detection model on the input tensor

    Args:
        session: An onnxruntime InferenceSession (must be laoded)
        input_tensor: Preprocessed image tesnor of shape (1, 3, H, W)

    Returns:
        heatmap: 2D float32 array of shape (H, W) with activation scores
    """
    # ONNX Runtime expects a dict {input_name: array}
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})

    # outputs is a list; we extract and squeeze to get (H, W)
    heatmap = outputs[0].squeeze(0).squeeze(0).astype(np.float32)

    return heatmap