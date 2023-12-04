import torch
import onnxruntime as ort
import keras

import numpy
from torchvision import models
from torchvision import transforms
from PIL import Image
from app.config import (
    IMAGE_PATH_PREFIX,
    ONNX_MODEL_PATH,
    TORCH_MODEL_PATH,
    KERAS_MODEL_PATH
)

import functools


def _get_default_torch_model() -> torch.nn.Module:
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


@functools.lru_cache()
def _get_torch_model() -> torch.nn.Module:
    model = _get_default_torch_model()
    weights = torch.load(TORCH_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.eval()
    return model


@functools.lru_cache()
def _get_onnx_model() -> ort.InferenceSession:
    return ort.InferenceSession(ONNX_MODEL_PATH)


@functools.lru_cache()
def _get_keras_model() -> keras.Model:
    model = keras.models.load_model(KERAS_MODEL_PATH, safe_mode=False)
    return model


def _get_transformed_image(image_path: str) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(Image.open(image_path))
    return image.unsqueeze(0)


def _torch_predict(model: torch.nn.Module, image_path: str) -> float:
    image = _get_transformed_image(image_path)
    with torch.no_grad():
        prediction = model(image)

    predicted_label = torch.argmax(prediction).item()
    return predicted_label


def _onnx_predict(model: ort.InferenceSession, image_path: str) -> float:
    image = _get_transformed_image(image_path)
    input_name = model.get_inputs()[0].name
    prediction = model.run(None, {input_name: image.numpy()})
    
    predicted_label = numpy.argmax(prediction).item()
    return predicted_label


def _keras_predict(model: keras.Model, image_path: str) -> float:
    image = _get_transformed_image(image_path)
    prediction = model.predict(image.numpy())
    
    predicted_label = numpy.argmax(prediction).item()
    return predicted_label


def load_models():
    _get_torch_model()
    _get_onnx_model()
    _get_keras_model()


def predict(model_type: str, image_path: str) -> float:
    
    handled_image_path = IMAGE_PATH_PREFIX + image_path

    match model_type:
        case 'torch':
            model = _get_torch_model()
            prediction = _torch_predict(model, handled_image_path)
        case 'onnx':
            model = _get_onnx_model()
            prediction = _onnx_predict(model, handled_image_path)
        case 'keras':
            model = _get_keras_model()
            prediction = _keras_predict(model, handled_image_path)
        case _:
            raise ValueError(f'Unknown model type: {model_type}')

    return prediction
        

