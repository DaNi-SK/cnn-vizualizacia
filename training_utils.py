import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import MNISTCNN


def train_model(
    epochs=5,
    learning_rate=0.001,
    batch_size=64,
    progress_callback=None,
):
    device = torch.device("cpu")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "backprop": None,
    }

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zaznamenáme prvý tréningový krok na vizualizáciu
            capture_backprop = history["backprop"] is None

            if capture_backprop:
                sample_image = images[0].detach().clone()
                sample_label = int(labels[0].item())

                # Váhy pred aktualizáciou optimalizátorom
                weight_before = (
                    model.features[0].weight[0, 0, :, :]
                    .detach()
                    .clone()
                )

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            if capture_backprop:
                # Gradienty prvého konvolučného filtra po backpropagation
                gradients = (
                    model.features[0].weight.grad[0, 0, :, :]
                    .detach()
                    .cpu()
                    .clone()
                )

                # Zaznamenáme aj chybu a výstupy siete
                history["backprop"] = {
                    "image": sample_image.cpu(),
                    "label": sample_label,
                    "prediction": int(outputs[0].argmax().item()),
                    "loss": float(loss.item()),
                    "weight_before": weight_before.cpu(),
                    "gradients": gradients,
                }

            optimizer.step()

            if capture_backprop:
                # Váhy po aktualizácii optimalizátorom Adam
                history["backprop"]["weight_after"] = (
                    model.features[0].weight[0, 0, :, :]
                    .detach()
                    .cpu()
                    .clone()
                )

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        model.eval()

        test_total_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_total_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        test_loss = test_total_loss / test_total
        test_accuracy = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, history)

    return model, history