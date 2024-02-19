"""
Este script agrupa los scripts de data_setup.py, model.py y engine.py para realizar el fine-tuning del modelo y guardarlo en models
"""
import torch
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torchmetrics import Accuracy
import os
import engine, data_setup, model


# Estableciendo los hyperparámetros
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 5


# Path a los datos
TRAIN_DIR = "data/dataset/train"
TEST_DIR = "data/dataset/test"


# Directorio de salida
OUTPUT_MODE_DIR = "models/"




if __name__ == "__main__":
    # Creando código agnóstico que aproveche la GPU en caso de que haya
    device = "cuda" if torch.cuda.is_available() else "cpu"



    """
    Estableciendo el transforms, train & test dataloaders, modelo, función de pérdida,
    optimizador, escalador y función de precisión
    """
    data_transforms = transforms.Compose([
        transforms.Resize(size= (224, 224)),
        transforms.ToTensor(),
    ])

    train_dataloader, test_dataloader, classes = data_setup.crear_dataloaders(
        train_dir = TRAIN_DIR,
        test_dir= TEST_DIR,
        transforms = data_transforms,
        batch_size = BATCH_SIZE
        )

    classificationModel = model.cargar_ResNet50(len(classes)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        classificationModel.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)

    scaler = GradScaler()

    accuracy_fn = Accuracy(task="multiclass", num_classes=len(classes))



    # Entrenando el modelo
    results = engine.train(
        model = classificationModel,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        loss_function = loss_fn,
        optimizer = optimizer,
        accuracy_fn = accuracy_fn,
        scaler = scaler,
        num_epochs = EPOCHS,
        device = device
    )

    # Guardando el modelo en la carpeta "models"
    try:
        listdir = os.listdir(OUTPUT_MODE_DIR)
        new_numeration = int(sorted(listdir)[-1][-4]) + 1
        new_name = f"OrangeNet_{new_numeration}.pt"
    except IndexError:
        new_name = "OrangeNet_0.pt"

    torch.save(classificationModel.state_dict(), f"{OUTPUT_MODE_DIR}/{new_name}")