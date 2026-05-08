import streamlit as st
import requests
import io
import os
from PIL import Image

API_URL = "http://127.0.0.1:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_COLORS = {
    "Black rot"                          : "#A32D2D",
    "Esca (Black Measles)"               : "#854F0B",
    "Leaf blight (Isariopsis Leaf Spot)" : "#185FA5",
    "healthy"                            : "#3B6D11",
}

CLASS_BG = {
    "Black rot"                          : "#FCEBEB",
    "Esca (Black Measles)"               : "#FAEEDA",
    "Leaf blight (Isariopsis Leaf Spot)" : "#E6F1FB",
    "healthy"                            : "#EAF3DE",
}

SAMPLE_IMAGES = {
    "Black rot"   : os.path.join(BASE_DIR, "test_images", "Black_rot.jpg"),
    "Esca"        : os.path.join(BASE_DIR, "test_images", "Esca.jpg"),
    "Leaf blight" : os.path.join(BASE_DIR, "test_images", "Leaf_blight.jpg"),
    "Healthy"     : os.path.join(BASE_DIR, "test_images", "Healthy.jpg"),
}


def call_api(image_bytes, filename):
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (filename, image_bytes, "image/jpeg")}
    )
    return response.json()


st.set_page_config(
    page_title="Grape Disease Classifier",
    page_icon="🍇",
    layout="wide"
)

st.title("🍇 Grape Disease Classifier")
st.caption("Upload a grape leaf image to detect disease")
st.divider()

col1, col2, col3 = st.columns([1, 1.5, 1])


with col1:
    st.subheader("Upload")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.caption("Or try a sample:")
    sample_cols = st.columns(2)
    for i, (label, path) in enumerate(SAMPLE_IMAGES.items()):
        with sample_cols[i % 2]:
            if st.button(label, use_container_width=True):
                with open(path, "rb") as f:
                    st.session_state["sample_bytes"] = f.read()
                    st.session_state["sample_name"]  = os.path.basename(path)

    st.divider()
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

# Resolve image source
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image_name  = uploaded_file.name
    image       = Image.open(uploaded_file)
elif "sample_bytes" in st.session_state:
    image_bytes = st.session_state["sample_bytes"]
    image_name  = st.session_state["sample_name"]
    image       = Image.open(io.BytesIO(image_bytes))
else:
    image_bytes = None
    image_name  = None
    image       = None

with col2:
    
    st.subheader("Image")
    if image:
        st.image(image, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="border:2px dashed #ccc; border-radius:12px; height:350px;
                        display:flex; align-items:center; justify-content:center; color:#aaa;">
                <p style="font-size:14px;">No image uploaded</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col3:

    st.subheader("Result")

    if image and predict_clicked:
        with st.spinner("Analyzing..."):
            result = call_api(image_bytes, image_name)

        pred  = result["prediction"]
        conf  = result["confidence"]
        color = CLASS_COLORS.get(pred, "gray")
        bg    = CLASS_BG.get(pred, "#f5f5f5")

        st.markdown(
            f"""
            <div style="background:{bg}; border-radius:12px; padding:20px; text-align:center; margin-bottom:16px;">
                <p style="font-size:11px; color:{color}; margin:0; letter-spacing:0.06em; text-transform:uppercase;">Prediction</p>
                <p style="font-size:20px; font-weight:500; color:{color}; margin:6px 0;">{pred}</p>
                <p style="font-size:14px; color:{color}; margin:0;">Confidence: {conf}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("Probability per class")
        for cls, prob in result["probabilities"].items():
            short = cls.split("(")[0].strip() if "(" in cls else cls
            color = CLASS_COLORS.get(cls, "gray")
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:6px 10px; border-radius:8px; margin-bottom:6px;
                            background:{'#f0f0f0' if cls != pred else bg};">
                    <span style="font-size:13px; color:{'#333' if cls != pred else color};">{short}</span>
                    <span style="font-size:13px; font-weight:500; color:{'#333' if cls != pred else color};">{prob}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif not image:
        st.markdown(
            """
            <div style="border:2px dashed #ccc; border-radius:12px; height:350px;
                        display:flex; align-items:center; justify-content:center; color:#aaa;">
                <p style="font-size:14px;">Results will appear here</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Click Predict to analyse the image")