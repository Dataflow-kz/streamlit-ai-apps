import streamlit as st
import requests
from openai import OpenAI

def generate_image_from_text(prompt, size, quality, api_key):
    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        n=1
    )
    return response.data[0].url

def image_from_url(url):
    img = requests.get(url)
    return img.content

st.set_page_config(
    page_title="DALL-E 3 Text-to-Image Generator", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🖼️"
)

def display_images(img_url, label):
    image = image_from_url(img_url)
    st.image(image, use_column_width=True, caption=label, output_format="PNG")
    st.download_button("Download Image", image, f"{label}.png", "png")

def main():
    st.title("🖼️ DALL-E 3 Text-to-Image Generator")
    st.markdown('''
                Generate stunning images from text descriptions using the DALL-E 3 model from OpenAI.

                **Reference:** [OpenAI DALL-E 3](https://openai.com/dall-e-3)

                ---
        ''')

    # Sidebar settings
    with st.sidebar:
        # Header
        st.image("https://github.com/silvermete0r/31github/blob/master/media/logo.png?raw=true", use_column_width=True)

        # Settings
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        image_size = st.selectbox("Select image size", ["1024x1024", "1792x1024", "1024x1792"])
        image_quality = st.selectbox("Select image quality", ["standard", "hd"])

        # Footer
        st.caption('Made with ❤️ by [DataFlow](https://dataflow.kz) team.')
        
    # Main prompt input & generate button
    prompt = st.text_area("Enter text prompt", "", height=100, max_chars=500, help="Enter a text prompt to generate an image.")

    if st.button("Generate Image"):
        if prompt.strip():
            with st.spinner("Generating image..."):
                try:
                    image = generate_image_from_text(prompt, image_size, image_quality, api_key)
                    label = f"Generated image from the prompt: {prompt}"
                except Exception as e:
                    st.warning(f"An error occurred: {e}")
                    image = 'https://astanatimes.com/wp-content/uploads/2021/02/Vangelia-2048x1519.jpg'
                    label = "Vangelia from Belovodiye. Photo credits: Elena Khardina. Source: astanatimes.com"

                if image:
                    display_images(image, label)
        else:
            st.warning("Please enter a text prompt.")

if __name__ == "__main__":
    main()
