from torchvision import transforms, models
from numpy import ndarray
from PIL import Image

import io
import numpy as np

def load_model():
    return models.resnet18(weights='DEFAULT')

def get_preprocessor():
    return transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_image(contents: bytes) -> Image.Image:
    bytes_io = io.BytesIO(contents)
    image = Image.open(bytes_io).convert("RGB")
    return image

def most_probability_class(x):
    x = softmax(x)
    idx = np.argmax(x)
    return int(idx)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()