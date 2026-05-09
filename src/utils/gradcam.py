# src/utils/gradcam.py


import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model, model_name="densenet121"):
    """
    Return the target convolutional layer for Grad-CAM based on model architecture.
    The last convolutional layer is used as it contains the highest level features.

    Parameters:
        model:      the loaded PyTorch model
        model_name: architecture name to select the correct layer

    Returns:
        list containing the target layer
    """
    if model_name == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
    elif model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]


def generate_gradcam(model, tensor, class_idx, model_name="densenet121"):
    """
    Generate a Grad-CAM++ heatmap for the predicted class.
    Grad-CAM++ is used over standard Grad-CAM as it produces better
    localization for multiple small regions like disease spots.

    Parameters:
        model:      the loaded PyTorch model
        tensor:     preprocessed input image tensor of shape (1, 3, 224, 224)
        class_idx:  predicted class index to generate heatmap for
        model_name: architecture name to select the correct target layer

    Returns:
        grayscale_cam: 2D numpy array of shape (224, 224) with activation values
    """
    target_layers = get_target_layer(model, model_name)

    cam           = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets       = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)

    # Remove batch dimension
    grayscale_cam = grayscale_cam[0]

    return grayscale_cam


def overlay_gradcam(original_image, grayscale_cam):
    """
    Overlay the Grad-CAM heatmap on the original image.
    Red areas indicate regions the model focused on most when making its prediction.

    Parameters:
        original_image: PIL image of the original leaf
        grayscale_cam:  2D numpy array with Grad-CAM activation values

    Returns:
        visualization: RGB numpy array of shape (224, 224, 3) with heatmap overlay
    """

    # Resize image to model input size and normalize to 0.0-1.0 range
    rgb_image     = np.array(original_image.resize((224, 224))) / 255.0
    rgb_image     = rgb_image.astype(np.float32)

    # Blend heatmap with original image
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    return visualization