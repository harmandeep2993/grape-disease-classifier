from torchvision import transforms


def get_inference_transforms():
    """
    Return the image preprocessing pipeline for inference.

    No augmentation is applied during inference. Only resizing and
    normalization are used to match the preprocessing applied during training.

    Normalization values are the ImageNet mean and standard deviation,
    which must match what was used during training since the model was
    pretrained on ImageNet.

    Returns:
        transforms.Compose: sequential pipeline of image transforms
    """
    return transforms.Compose([

        # Resize to 224x224 as expected by DenseNet121, ResNet50 and EfficientNet-B0
        transforms.Resize((224, 224)),

        # Convert PIL image to tensor and scale pixel values from 0-255 to 0.0-1.0
        transforms.ToTensor(),

        # Normalize using ImageNet mean and std to match training preprocessing
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])