"""
Este script contiene la clases pipeline que se utiliza para la realización de predicciones
"""
import numpy as np
from PIL import Image
from torchvision import transforms
import model
import torch
import os
import json


class OrangePipeline():
    """
    Esta clase se utiliza para la realización de inferencias

    Args de entrada del constructor:
        model_path: Es el nombre del directorio en el que están los modelos.
        labels_json_path: Es el path en el que está el json con los labels para clasificar con el nombre.
    """
    def __init__(self,
                 model_path : str = "models",
                 labels_json_path : str = "labels.json"
                 ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        latest_model_state_dict_path = sorted(os.listdir(model_path))[-1]
        self.orange_model = model.cargar_ResNet50().to(self.device)
        self.orange_model.load_state_dict(torch.load(f"models/{latest_model_state_dict_path}", map_location=torch.device(self.device)))
        self.data_transforms = transforms.Compose([
            transforms.Resize(size= (224, 224)),
            transforms.ToTensor(),
        ])

        with open(labels_json_path, "r") as f:
            self.classification_labels = json.load(f)
    
    def inference(self, img : any) -> dict:
        """
        Este método se utiliza para la inferencia de imágenes

        Args:
            img: Recibe una imágen, ya sea como un np.ndarray, PIL.Image o incluso el path hacia una imagen.
        
        Devuelve:
            Un diccionario con las probabilidades de que la imagen pertenezca a cada una de las clases.
        """
        input_img_data_type = type(img)
        if input_img_data_type == str:
            img = Image.open(img)
        
        elif input_img_data_type == np.ndarray:
            img = Image.fromarray(img)

        tensor_img = self.data_transforms(img)
        return self.__inference(tensor_img)

    def __inference(self, img_tensor: torch.Tensor) -> dict:
        self.orange_model.eval()
        with torch.inference_mode():
            logits = self.orange_model(img_tensor.unsqueeze(0).to(self.device))

        pred_probs = torch.softmax(logits, dim = 1)[0]
        return {label : round(pred_probs[index].item(), 4) for index, label in enumerate(self.classification_labels["classes"])}