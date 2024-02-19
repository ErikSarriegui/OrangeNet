import torch
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torchmetrics import Accuracy
import os
import engine, data_setup, model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = transforms.Compose([
        transforms.Resize(size= (224, 224)),
        transforms.ToTensor(),
    ])

    train_dataloader, test_dataloader, classes = data_setup.crear_dataloaders(
        train_dir = "data/dataset/train",
        test_dir= "data/dataset/test",
        transforms = data_transforms,
        batch_size = 32)

    classificationModel = model.cargar_ResNet50(len(classes)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        classificationModel.parameters(),
        lr=0.001,
        weight_decay=0.0001)

    scaler = GradScaler()

    accuracy_fn = Accuracy(task="multiclass", num_classes=len(classes))

    # Train the model
    results = engine.train(
        model = classificationModel,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        loss_function = loss_fn,
        optimizer = optimizer,
        accuracy_fn = accuracy_fn,
        scaler = scaler,
        num_epochs = 1,
        device = device
    )

    # Save the model
    try:
        listdir = os.listdir("models/")
        new_numeration = int(sorted(listdir)[-1][-4]) + 1
        new_name = f"OrangeNet_{new_numeration}.pt"
    except IndexError:
        new_name = "OrangeNet_0.pt"

    torch.save(classificationModel.state_dict(), f"models/{new_name}")