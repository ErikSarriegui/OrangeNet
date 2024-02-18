"""
Este script contiene un método que devuelve una instancia del modelo que vamos a utilizar (ResNet50)
"""

import torchvision
import torch
from torch import nn

def cargar_ResNet50(out_features : int = 4):
  """
  Esta función crea una instancia del modelo que se va a utilizar.

  Para hacer esto, coge como argumento los class_names del dataset.

  Args:
    class_names : es una lista con las classes a predecir

  Devuelve:
    Una instancia de ResNet50 pre-entrenada con una cabeza no
    no entrenada para el fine-tuning

  Ejemplo de uso:
    model = cargar_ResNet50(dataset_class_names)
  """
  model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
  # Congelar las capas convolucionales del modelo
  for param in model.parameters():
      if isinstance(param, nn.Conv2d):
          param.requires_grad = False

  # Modificar la cabeza del modelo para el fine-tuning
  num_features = model.fc.in_features

  # Añadir capas
  model.fc = nn.Sequential(
      nn.Linear(num_features, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, out_features)
  )

  return model