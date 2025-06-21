# Local Image Background Eraser

A Streamlit app to remove image backgrounds locally using either OpenCV GrabCut or U2Net (ONNX).

## Features
- Remove backgrounds from images using:
  - **U2Net (ONNX)** (high quality, requires model file)
  - **OpenCV + GrabCut** (no model required)
- Download the processed image directly from the app.

## Installation
1. Clone this repository or download the source code.
2. Install Python 3.8 or higher.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Download U2Net ONNX Model
Download the `u2net.onnx` model file from [here](https://huggingface.co/tomjackson2023/rembg/blob/main/u2net.onnx) and place it in the project directory (or as required by your `u2net_onnx` module).

## Run the App
```bash
streamlit run app.py
```

Open the provided local URL in your browser to use the app.

---

**Note:**
- For best results with U2Net, ensure the ONNX model is downloaded before running the app.
- Supported image formats: jpg, jpeg, png, jfif.
