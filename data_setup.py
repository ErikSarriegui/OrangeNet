"""
Este script contiene una función para crear los DataLoaders con los que se entrena el modelo
"""

import torch
import torchvision
import os

NUM_WORKERS = os.cpu_count()

def crear_dataloaders(
    train_dir: str,
    test_dir: str,
    transforms: torchvision.transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
) -> tuple:
  """
  Esta función crea los DataLoaders, tanto de entrenamiento como de testeo.

  Para hacer esto, coge como argumento los directorios del dataset, los carga
  como Datasets para luego cambiarlos a DataLoaders.

  Args:
    train_dir: Path al directorio de entrenamiento.
    test_dir: Path al directorio de testeo.
    transforms: torchvision transforms a hacer en los datos de entrenamiento
              y testeo.
    batch_size: Tamaño de los batches del DataLoader.
    num_workers: El número de workers. Por defecto es el número de cpus 
    disponible.

  Devuelve:
    Una tupla con (train_dataloader, test_dataloader, class_names), en el que
    class_names es una lista con las classes a predecir.

  Ejemplo de uso:
    train_dataloader, test_dataloader, class_names = create_dataloaders(
      train_dir=path/to/train_dir,
      test_dir=path/to/test_dir,
      transforms=some_transform,
      batch_size=32,
      num_workers=4)
  """
  train_dataset = torchvision.datasets.ImageFolder(train_dir, transform = transforms)

  test_dataset = torchvision.datasets.ImageFolder(test_dir, transform = transforms)

  train_dataloader = torch.utils.data.DataLoader(
      dataset = train_dataset,
      batch_size = batch_size,
      shuffle = True,
      num_workers = num_workers)
  
  test_dataloader = torch.utils.data.DataLoader(
      dataset = test_dataset,
      batch_size = batch_size,
      shuffle = False,
      num_workers = num_workers)
  
  return train_dataloader, test_dataloader, train_dataset.classes