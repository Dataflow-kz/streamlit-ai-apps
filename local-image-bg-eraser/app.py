import streamlit as st
from opencv_grabcut import remove_background_grabcut
from u2net_onnx import remove_background_u2net
import time
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title="Local Image Background Eraser", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧹"
)

st.title("Local Image Background Eraser")

method = st.selectbox(
    "Select a method to remove background",
    options=["U2Net (ONNX)", "OpenCV + GrabCut"],
    index=0
)

with st.expander("Methods Description", expanded=False):
    st.markdown("""
Method	| Strengths	| Weaknesses | Link
---|---|---|---
U²-Net ONNX | High quality, runs offline	| Requires downloading model + setup | [U2Net](https://arxiv.org/abs/2212.06466)
OpenCV + GrabCut | No model needed, works offline | Struggles with complex backgrounds | [OpenCV GrabCut](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html)
    """)

image_object = st.file_uploader(
    "Upload an image to remove its background",
    type=["jpg", "jpeg", "png", "jfif"],
    accept_multiple_files=False
)

if image_object is not None:
    if method == "OpenCV + GrabCut":
        startTime = time.time()
        with st.spinner("Processing with OpenCV + GrabCut..."):
            before, after = st.columns(2)
            with before:
                st.image(image_object, caption="Original Image", use_container_width=True)
            with after:
                output_image = remove_background_grabcut(image_object)
                st.image(output_image, caption="Background Removed", use_container_width=True)
            endTime = time.time()
            processing_time = endTime - startTime
            st.success("Background removed using OpenCV + GrabCut. Completed in {:.2f} seconds.".format(processing_time))
            img = Image.fromarray(output_image.astype('uint8'))
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            download_button = st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="background_removed.png",
                mime="image/png"
            )

    elif method == "U2Net (ONNX)":
        startTime = time.time()
        with st.spinner("Processing with U2Net (ONNX)..."):
            before, after = st.columns(2)
            with before:
                st.image(image_object, caption="Original Image", use_container_width=True)
            with after:
                output_image = remove_background_u2net(image_object)
                st.image(output_image, caption="Background Removed", use_container_width=True)
            endTime = time.time()
            processing_time = endTime - startTime
            st.success("Background removed using U2Net (ONNX). Completed in {:.2f} seconds.".format(processing_time))
            img = Image.fromarray(output_image.astype('uint8'))
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            download_button = st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="background_removed.png",
                mime="image/png"
            )