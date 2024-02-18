import numpy as np
from PIL import Image
from torchvision import transforms
import model
import torch
import os
import json

# Loading the Model
latest_model_state_dict = sorted(os.listdir("models"))[-1]
classificationModel = model.cargar_ResNet50()
classificationModel.load_state_dict(torch.load(f"models/{latest_model_state_dict}"))

# Loading labels
with open("labels.json", "r") as f:
    labels = json.load(f)

data_transforms = transforms.Compose([
    transforms.Resize(size= (224, 224)),
    transforms.ToTensor(),
])

def inference_with_path(img_path : str, model = classificationModel, transforms = data_transforms, labels = labels):
    img = Image.open(img_path) 
    img_tensor = transforms(img)
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor.unsqueeze(0))

    pred_probs = torch.softmax(logits, dim = 1)[0]

    return {label : round(pred_probs[index].item(), 6) for index, label in enumerate(labels["classes"])}

def inference(img : Image, model = classificationModel, transforms = data_transforms, labels = labels):
    img_tensor = transforms(img)
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor.unsqueeze(0))

    pred_probs = torch.softmax(logits, dim = 1)[0]

    return {label : round(pred_probs[index].item(), 6) for index, label in enumerate(labels["classes"])}