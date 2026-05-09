# src/models/model.py


import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name="densenet121", num_classes=4):
    """
    Build a pretrained CNN model with a custom classifier head.

    The backbone is loaded without pretrained weights since we load
    our own fine-tuned weights separately via load_model. The classifier
    head is replaced with a dropout layer followed by a linear layer
    that outputs one score per disease class.

    Parameters:
        model_name:  architecture to build. one of densenet121, resnet50, efficientnet_b0
        num_classes: number of output classes. 4 for our grape disease dataset

    Returns:
        model: PyTorch model with custom classifier head, weights not loaded yet
    """

    if model_name == "densenet121":
        model       = models.densenet121(weights=None)
        in_features = model.classifier.in_features

        # Replace classifier with dropout + linear for 4 class output
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":
        model       = models.resnet50(weights=None)
        in_features = model.fc.in_features

        # ResNet uses model.fc as the final layer
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model       = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features

        # EfficientNet classifier is a Sequential so index [1] is the linear layer
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    return model


def load_model(model_path="models/densenet121_finetuned_best.pth", model_name="densenet121"):
    """
    Build the model architecture and load fine-tuned weights from disk.

    Automatically selects GPU if available, otherwise falls back to CPU.
    Sets model to evaluation mode to disable dropout during inference.

    Parameters:
        model_path:  path to the saved .pth weights file
        model_name:  architecture name passed to build_model

    Returns:
        model:  PyTorch model loaded with fine-tuned weights, ready for inference
        device: torch.device indicating where the model is running (cuda or cpu)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build empty model structure
    model = build_model(model_name)

    # Load fine-tuned weights and map to available device
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    return model, device