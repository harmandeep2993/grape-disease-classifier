# api/main.py


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
import base64
import io
from PIL import Image
from src.models.predict import predict


# Initialize FastAPI application
app = FastAPI(
    title="Grape Disease Classifier",
    description="REST API for classifying grape leaf diseases using DenseNet121 with transfer learning.",
    version="1.0.0"
)


@app.get("/health")
def health():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded grape leaf image and returns disease classification results.

    Parameters:
        file: JPG or PNG image of a grape leaf

    Returns:
        prediction:    predicted disease class name
        confidence:    prediction confidence as a percentage
        probabilities: confidence score for each of the 4 classes
        spot_count:    number of disease spots detected by OpenCV
        annotated:     base64 encoded PNG with disease spots outlined in red
        gradcam:       base64 encoded PNG showing model attention heatmap
    """

    # Save uploaded file temporarily to disk for processing
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model prediction, disease spot detection and Grad-CAM generation
    result = predict(temp_path)

    # Clean up temporary file after prediction is complete
    os.remove(temp_path)

    # Encode annotated disease spot image to base64 for JSON transport
    annotated_buf = io.BytesIO()
    result["annotated"].save(annotated_buf, format="PNG")
    annotated_b64 = base64.b64encode(annotated_buf.getvalue()).decode()

    # Encode Grad-CAM heatmap image to base64 for JSON transport
    gradcam_buf = io.BytesIO()
    result["gradcam"].save(gradcam_buf, format="PNG")
    gradcam_b64 = base64.b64encode(gradcam_buf.getvalue()).decode()

    return JSONResponse(content={
        "prediction"    : result["prediction"],
        "confidence"    : result["confidence"],
        "probabilities" : result["probabilities"],
        "spot_count"    : result["spot_count"],
        "annotated"     : annotated_b64,
        "gradcam"       : gradcam_b64,
    })