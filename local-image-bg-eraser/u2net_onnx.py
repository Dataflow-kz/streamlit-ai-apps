import onnxruntime as ort
import cv2
import numpy as np
from io import BytesIO
import sys
import os

try:
    sess = ort.InferenceSession("models/u2net.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
except Exception as e:
    print(f"Failed to load ONNX model: {e}", file=sys.stderr)
    sess = None
    input_name = None

def preprocess(img):
    img = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, 0)

def postprocess(prediction, original_shape):
    # U²-Net returns a tuple of 6 outputs; we use the first (most accurate)
    pred = prediction[0]  # shape: (1, 1, 320, 320)
    pred = np.squeeze(pred)  # shape: (320, 320)
    pred = cv2.resize(pred, (original_shape[1], original_shape[0]))
    pred = (pred * 255).astype(np.uint8)  # convert to 0–255 uint8
    mask = (pred > 128).astype(np.uint8) * 255  # binarize
    return mask

def remove_background_u2net(image):
    if sess is None or input_name is None:
        raise RuntimeError("U2Net ONNX model is not loaded. Check model path and dependencies.")

    # Read the image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {image}")
    elif hasattr(image, 'read'):
        image_bytes = image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image.seek(0)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Unsupported image input type.")

    input_tensor = preprocess(img)

    try:
        pred = sess.run(None, {input_name: input_tensor})
    except Exception as e:
        raise RuntimeError(f"ONNX inference failed: {e}") from e

    mask = postprocess(pred, img.shape[:2])

    # Ensure mask is uint8 and shape matches
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    assert mask.dtype == np.uint8, f"Mask must be uint8 but got {mask.dtype}"
    assert mask.shape == img.shape[:2], f"Mask shape {mask.shape} must match image shape {img.shape[:2]}"

    # Convert BGR to RGBA
    b, g, r = cv2.split(img)
    alpha = mask
    rgba = cv2.merge((r, g, b, alpha))

    return rgba
