"""
Este script contiene tres métodos: train_step, test_step y train.

train_step : Consiste en el loop de entrenamiento, los batches se iteran dentro
de este pero los epochs no.

test_step : Consiste en el loop de testeo, los batches se iteran dentro
de este pero los epochs no.

train : Agrupa ambas funciones para realizar un entrenamiento completo. Itera
los epochs dentro de él mismo.
"""
import torch
from torch import nn
import torchmetrics
from torch.cuda.amp import autocast
from torchmetrics import Accuracy
from tqdm.auto import tqdm

def train_step(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    loss_function: nn.Module,
    accuracy_fn: torchmetrics.Accuracy,
    scaler: torch.cuda.amp,
    train_batches_loss: list,
    train_batches_acc: list,
    device: str
    ) -> None:
    """
    Esta función itera una vez sobre todos los batches del DataLoader
    de entrenamiento, hace el forward pass, el backpropagation y
    actualiza los pesos.

    Args:
        model: El modelo a entrenar.
        train_dataloader: El DataLoader con los datos de entrenamiento.
        optimizer: El optimizador.
        loss_function: La función de pérdida.
        accuracy_fn: La función de precisión, siempre Accuracy de
        torchmetrics.
        scaler: El scaler.
        train_batches_loss: Un lista que se llenará con el loss de cada
        uno de los batches, pensado para hacer plots.
        train_batches_acc: Un lista que se llenará con la precisión del
        modelo en cada sobre de los batches, pensado para hacer plots.
        device: El dispositivo (cpu, cuda) sobre el que se realizará
        el entrenamiento.

    Devuelve:
        Las listas de train_batches_loss y train_batches_acc. Para poder
        hacer plots con ellos.

    Ejemplo de uso:
        Ver dentro del método train()
    """

    epoch_loss, epoch_acc = 0, 0

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        model.train()
        data, targets = data.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Enable autocasting for mixed precision
        with autocast():
            outputs = model(data)
            loss = loss_function(outputs, targets)
        
        preds = outputs.argmax(dim = 1)
        acc = accuracy_fn(preds.cpu(), targets.cpu())
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # Perform backward pass and gradient scaling
        scaler.scale(loss).backward()

        # Update model parameters
        scaler.step(optimizer)
        scaler.update()

    train_batches_acc.append(epoch_loss / len(train_dataloader))
    train_batches_acc.append(epoch_acc / len(train_dataloader))





def test_step(
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    accuracy_fn: torchmetrics.Accuracy,
    test_batches_loss: list,
    test_batches_acc: list,
    device: str
    ) -> None:
    """
    Esta función itera una vez sobre todos los batches del DataLoader de
    testeo, hace el forward pass para determinar la pérdida y la precisión.

    Args:
        model: El modelo a entrenar.
        test_dataloader: El DataLoader con los datos de testeo.
        loss_function: La función de pérdida.
        accuracy_fn: La función de precisión, siempre Accuracy de
        torchmetrics.
        test_batches_loss: Un lista que se llenará con el loss de cada
        uno de los batches, pensado para hacer plots.
        test_batches_acc: Un lista que se llenará con la precisión del
        modelo en cada sobre de los batches, pensado para hacer plots.
        device: El dispositivo (cpu, cuda) sobre el que se realizará
        el entrenamiento.

    Devuelve:
        Las listas de test_batches_loss y test_batches_acc. Para poder
        hacer plots con ellos.

    Ejemplo de uso:
        Ver dentro del método train()
    """
    epoch_loss, epoch_acc = 0, 0
  
    for data, targets in test_dataloader:
        data, targets = data.to(device), targets.to(device)
        model.eval()
        
        with torch.inference_mode():
            logits = model(data)
            loss = loss_function(logits, targets)
        
        preds = logits.argmax(dim = 1)
        acc = accuracy_fn(preds.cpu(), targets.cpu())

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    test_batches_loss.append(epoch_loss / len(test_dataloader))
    test_batches_acc.append(epoch_acc / len(test_dataloader))




def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim,
    accuracy_fn: torchmetrics.Accuracy,
    scaler: torch.cuda.amp,
    num_epochs: int,
    device: str
    ):
    """
    Esta función agrupa tanto la fución de train_step como la
    de test_setp para realizar un entrenamiento y testeo
    conjunto en una misma función.

    Args:
        model: El modelo a entrenar.
        train_dataloader: El DataLoader con los datos de entrenamiento.
        test_dataloader: El DataLoader con los datos de testeo.
        loss_function: La función de pérdida.
        optimizer: El optimizador.
        accuracy_fn: La función de precisión, siempre Accuracy de
        torchmetrics.
        scaler: El scaler.
        num_epochs: El número de epochs con los que se entrenará y
        testeará el modelo
        device: Dispositivo (cpu o cuda) en el que se realizará
        el entrenamiento y testeo

    Devuelve:
        Devuelve un dict con la pérdida y la precisión de todos los batches
        en todos los epochs. Esta pensado para hacer plots.
    """
    train_batches_loss, train_batches_acc = [], []
    test_batches_loss, test_batches_acc = [], []

    for epoch in tqdm(range(num_epochs)):
        train_step(
            model = model,
            train_dataloader = train_dataloader,
            optimizer = optimizer,
            loss_function = loss_function,
            accuracy_fn = accuracy_fn,
            scaler = scaler,
            train_batches_loss = train_batches_loss,
            train_batches_acc = train_batches_acc,
            device = device
            )
        
        test_step(
            model = model,
            test_dataloader = test_dataloader,
            loss_function = loss_function,
            accuracy_fn = accuracy_fn,
            test_batches_loss = test_batches_loss,
            test_batches_acc = test_batches_acc,
            device = device
            )
        
        if epoch % 1 == 0:
            print(f"{epoch}: Training Loss: {train_batches_loss[-1]} / Trainng Accuraccy: {train_batches_acc[-1]} || Testing Loss: {test_batches_loss[-1]} / Testing Accuraccy: {test_batches_acc[-1]}")
        
    return {
        "train_batches_loss" : train_batches_loss,
        "train_batches_acc" : train_batches_acc,
        "test_batches_loss" : test_batches_loss,
        "test_batches_acc" : test_batches_acc
    }