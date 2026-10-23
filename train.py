import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import MNISTCNN


# 1. Nastavenie experimentu
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cpu")
print("Výpočtové zariadenie:", device)


# 2. Príprava datasetu
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# 3. Vytvorenie modelu a nástrojov na učenie
model = MNISTCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# 4. Záznam výsledkov
history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}


# 5. Tréning
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpocha {epoch + 1}/{EPOCHS}")

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_index + 1) % 100 == 0:
            print(
                f"  Dávka {batch_index + 1}/{len(train_loader)} "
                f"| priebežná strata: {total_loss / total:.4f}"
            )

    train_loss = total_loss / total
    train_accuracy = correct / total

    # 6. Vyhodnotenie na testovacích dátach
    model.eval()

    test_total_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

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

    print(f"Tréningová strata: {train_loss:.4f}")
    print(f"Tréningová presnosť: {train_accuracy * 100:.2f} %")
    print(f"Testovacia strata: {test_loss:.4f}")
    print(f"Testovacia presnosť: {test_accuracy * 100:.2f} %")


print("\nTréning dokončený.")

with open("results.json", "w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)

print("Výsledky boli uložené do results.json")

torch.save(model.state_dict(), "model.pth")
print("Natrénovaný model bol uložený do model.pth")