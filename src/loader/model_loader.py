import onnxruntime as ort
import streamlit as st
from src.config import MODEL_PATH

@st.cache_resource
def load_session(model_path: str = str(MODEL_PATH)) -> ort.InferenceSession:
    """
    Load and cache an ONNX Runtime InferenceSession.
    
    Args:
        model_path: Path to the .onnx model file
    Returns:
        ort.InferenceSession: The loaded and cached session
    """
    # TODO: add provider fallbacks or session options
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])