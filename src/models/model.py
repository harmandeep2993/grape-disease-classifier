import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name="densenet121", num_classes=4):
    if model_name == "densenet121":
        model       = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet50":
        model       = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "efficientnet_b0":
        model       = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    return model


def load_model(model_path="models/densenet121_finetuned_best.pth", model_name="densenet121"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model  = model.to(device)
    model.eval()
    return model, device