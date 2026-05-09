#src/models/predict.py


import sys
import os

# Add project root to path so src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
from PIL import Image
from src.models.model import load_model
from src.data import get_inference_transforms
from src.utils.disease_detector import detect_disease_spots
from src.utils.gradcam import generate_gradcam, overlay_gradcam


# Class labels matching the order assigned by ImageFolder during training
CLASS_NAMES = [
    "Black rot",
    "Esca (Black Measles)",
    "Leaf blight (Isariopsis Leaf Spot)",
    "healthy"
]


def predict(image_path, model_path="models/densenet121_finetuned_best.pth", model_name="densenet121"):
    """
    Run full inference pipeline on a single leaf image.

    Parameters:
        image_path: path to the input image file
        model_path: path to the saved model weights (.pth file)
        model_name: architecture name used to build the correct model structure

    Returns:
        dictionary containing:
            prediction:    predicted disease class name
            confidence:    prediction confidence as a percentage
            probabilities: confidence score for each of the 4 classes
            spot_count:    number of disease spots detected
            annotated:     PIL image with disease spots outlined in red
            gradcam:       PIL image showing model attention heatmap
    """

    # Load model and move to available device (GPU or CPU)
    model, device = load_model(model_path, model_name)

    # Get inference transforms (resize, normalize, no augmentation)
    transform = get_inference_transforms()

    # Load image and convert to RGB to handle any RGBA or grayscale inputs
    image  = Image.open(image_path).convert("RGB")

    # Apply transforms and add batch dimension for model input
    tensor = transform(image).unsqueeze(0).to(device)

    # Run forward pass without computing gradients to save memory
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()

    # Detect and outline disease spots using OpenCV color thresholding
    annotated, spot_count, _ = detect_disease_spots(image)
    annotated_pil = Image.fromarray(annotated)

    # Generate Grad-CAM heatmap showing which regions the model focused on
    grayscale_cam = generate_gradcam(model, tensor, pred, model_name)
    gradcam_vis   = overlay_gradcam(image, grayscale_cam)
    gradcam_pil   = Image.fromarray(gradcam_vis)

    return {
        "prediction"    : CLASS_NAMES[pred],
        "confidence"    : round(probs[pred].item() * 100, 2),
        "probabilities" : {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
        "spot_count"    : spot_count,
        "annotated"     : annotated_pil,
        "gradcam"       : gradcam_pil,
    }