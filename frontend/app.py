import time
import base64
import streamlit as st
import requests
import io
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_URL  = os.getenv("API_URL", "http://127.0.0.1:8000")
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

MODEL_INFO = {
    "Architecture" : "DenseNet121",
    "Parameters"   : "7M",
    "Pretrained on": "ImageNet",
    "Fine-tuned on": "PlantVillage",
    "Test accuracy": "99.34%",
}


def call_api(image_bytes, filename):
    for attempt in range(3):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (filename, image_bytes, "image/jpeg")},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"error": str(e)}


st.set_page_config(
    page_title="Grape Disease Classifier",
    page_icon="🍇",
    layout="wide"
)

st.title("🍇 Grape Disease Classifier")
st.caption("Upload a grape leaf image to detect disease instantly")

# Metric cards at the top
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy",    "99.34%")
m2.metric("Training Images",  "4,062")
m3.metric("Disease Classes",  "4")
m4.metric("Model",            "DenseNet121")

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

    st.divider()
    st.caption("Disease classes")
    classes = {
        "Black rot"   : "#A32D2D",
        "Esca"        : "#854F0B",
        "Leaf blight" : "#185FA5",
        "Healthy"     : "#3B6D11",
    }
    for cls, color in classes.items():
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">'
            f'<div style="width:8px; height:8px; border-radius:50%; background:{color};"></div>'
            f'<span style="font-size:13px; color:var(--color-text-secondary);">{cls}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

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

# Call API before columns render
if image and predict_clicked:
    with st.spinner("Analyzing..."):
        result = call_api(image_bytes, image_name)
        st.session_state["result"] = result

if not image:
    st.session_state.pop("result", None)

with col2:
    st.subheader("Image")
    if image:
        has_result    = "result" in st.session_state
        has_annotated = has_result and "annotated" in st.session_state.get("result", {})
        has_gradcam   = has_result and "gradcam"   in st.session_state.get("result", {})

        if has_annotated or has_gradcam:
            tab1, tab2, tab3 = st.tabs(["Original", "Disease Spots", "Grad-CAM"])
            with tab1:
                st.image(image, use_container_width=True)
            with tab2:
                if has_annotated:
                    annotated_bytes = base64.b64decode(st.session_state["result"]["annotated"])
                    annotated_image = Image.open(io.BytesIO(annotated_bytes))
                    st.image(annotated_image, use_container_width=True)
                    spot_count = st.session_state["result"].get("spot_count", 0)
                    st.caption(f"Found {spot_count} disease spots outlined in red")
            with tab3:
                if has_gradcam:
                    gradcam_bytes = base64.b64decode(st.session_state["result"]["gradcam"])
                    gradcam_image = Image.open(io.BytesIO(gradcam_bytes))
                    st.image(gradcam_image, use_container_width=True)
                    st.caption("Red areas show where the model focused most")
        else:
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

    if "result" in st.session_state and image:
        result = st.session_state["result"]

        if "error" in result:
            st.error(f"API error: {result['error']}")
        else:
            pred  = result["prediction"]
            conf  = result["confidence"]
            color = CLASS_COLORS.get(pred, "gray")
            bg    = CLASS_BG.get(pred, "#f5f5f5")

            st.markdown(
                f"""
                <div style="background:{bg}; border-radius:12px; padding:16px; text-align:center; margin-bottom:12px;">
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
                        <span style="font-size:13px; color:{'#555' if cls != pred else color};">{short}</span>
                        <span style="font-size:13px; font-weight:500; color:{'#555' if cls != pred else color};">{prob}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    elif not image:
        st.markdown(
            """
            <div style="border:2px dashed #ccc; border-radius:12px; height:200px;
                        display:flex; align-items:center; justify-content:center; color:#aaa;">
                <p style="font-size:14px;">Results will appear here</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Click Predict to analyse the image")

    st.divider()

    # Model information section
    st.caption("Model info")
    for key, val in MODEL_INFO.items():
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; padding:4px 0;">
                <span style="font-size:12px; color:#888;">{key}</span>
                <span style="font-size:12px; font-weight:500; color:#333;">{val}</span>
            </div>
            """,
            unsafe_allow_html=True
        )