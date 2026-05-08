import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from PIL import Image
from src.models.model import load_model
from src.data.transforms import get_inference_transforms

CLASS_NAMES = [
    "Black rot",
    "Esca (Black Measles)",
    "Leaf blight (Isariopsis Leaf Spot)",
    "healthy"
]


def predict(image_path, model_path="models/densenet121_finetuned_best.pth"):
    model, device = load_model(model_path)
    transform     = get_inference_transforms()

    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()

    return {
        "prediction"    : CLASS_NAMES[pred],
        "confidence"    : round(probs[pred].item() * 100, 2),
        "probabilities" : {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/Healthy.jpg"
    result = predict(image_path)
    print(result)