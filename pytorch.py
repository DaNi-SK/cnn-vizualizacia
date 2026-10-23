import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=64):
    """
    Načíta MNIST dataset, aplikuje normalizáciu a vráti DataLoadery
    pre trénovaciu a testovaciu množinu.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Štandardná normalizácia pre MNIST
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# Konvolučná neurónová sieť
class CNN(nn.Module): # HOTOVO
    def __init__(self, num_layers=2, activation_name="ReLU"):
        super(CNN, self).__init__()  # HOTOVO
        
        # Výber aktivačnej funkcie
        if activation_name == "ReLU":
            self.act = nn.ReLU()
        elif activation_name == "Leaky ReLU":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        elif activation_name == "Sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Neznáma aktivačná funkcia: {activation_name}")
            
        layers = []
        in_channels = 1
        
        # Preddefinované počty filtrov pre jednotlivé bloky (16 -> 32 -> 64)
        filter_sizes = [16, 32, 64]
        current_size = 28  # Pôvodný rozmer MNIST obrázka je 28x28 pixelov
        
        # Dynamické skladanie konvolučných blokov
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            # Konvolučná vrstva zachovávajúca rozmer vďaka paddingu
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(self.act)
            # Max Pooling zmenší priestorový rozmer na polovicu
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
            current_size = current_size // 2  # Aktualizácia rozmeru (28 -> 14 -> 7 -> 3)
            
        self.features = nn.Sequential(*layers)
        
        # Výpočet rozmeru pre vstup do plne prepojenej vrstvy (Flatten)
        flatten_size = in_channels * current_size * current_size
        
        # Klasifikátor (Fully Connected vrstvy)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 64),
            self.act,
            nn.Linear(64, 10)  # 10 výstupných tried pre číslice 0 až 9
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x