import torch

from src.model import MNISTCNN


model = MNISTCNN()

# Vytvoríme jeden náhodný obrázok s rozmermi MNIST.
sample = torch.randn(1, 1, 28, 28)

# Pošleme obrázok cez neurónovú sieť.
output = model(sample)

print("Rozmer vstupu:", sample.shape)
print("Rozmer výstupu:", output.shape)
print("Výstupné hodnoty:", output)
