import numpy as np
from PIL import Image
from torchvision import transforms
import model
import torch
import os
import json

class OrangePipeline():
    def __init__(self, model_path = "models", labels_json_path = "labels.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        latest_model_state_dict_path = sorted(os.listdir(model_path))[-1]
        self.orange_model = model.cargar_ResNet50().to(self.device)
        self.orange_model.load_state_dict(torch.load(f"models/{latest_model_state_dict_path}", map_location=torch.device(self.device)))
        self.data_transforms = transforms.Compose([
            transforms.Resize(size= (224, 224)),
            transforms.ToTensor(),
        ])

        with open(labels_json_path, "r") as f:
            self.labels = json.load(f)
    
    def inference(self, img):
        data_type = type(img)
        if data_type == str:
            img = Image.open(img)
        
        if data_type == np.ndarray:
            img = Image.fromarray(img)

        tensor_img = self.data_transforms(img)
        return self.__inference(tensor_img)

    def __inference(self, img_tensor: torch.Tensor):
        self.orange_model.eval()
        with torch.inference_mode():
            logits = self.orange_model(img_tensor.unsqueeze(0).to(self.device))

        pred_probs = torch.softmax(logits, dim = 1)[0]
        return {label : round(pred_probs[index].item(), 4) for index, label in enumerate(self.labels["classes"])}