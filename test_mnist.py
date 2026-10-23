from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

images, labels = next(iter(loader))

print("Počet obrázkov v dávke:", images.shape[0])
print("Rozmer obrázkov:", images.shape)
print("Rozmer označení:", labels.shape)
print("Označenia prvých 10 obrázkov:", labels[:10].tolist())