import torch
from torchvision import datasets, transforms

from src.model import MNISTCNN


# 1. Načítanie natrénovaného modelu
model = MNISTCNN()

state_dict = torch.load(
    "model.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()


# 2. Načítanie jedného obrázka z MNIST
dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

image, label = dataset[0]


# 3. Príprava miesta na zachytenie výpočtov
activations = {}


def capture_features(module, inputs, output):
    activations["conv1"] = output.detach()


# 4. Zachytenie výstupov pred a po aktivácii ReLU
activations_after_relu = {}


def capture_after_relu(module, inputs, output):
    activations_after_relu["relu1"] = output.detach()


handle = model.features[0].register_forward_hook(capture_features)
handle_relu = model.features[1].register_forward_hook(capture_after_relu)



# 5. Spustenie modelu na obrázku
with torch.no_grad():
    output = model(image.unsqueeze(0))


# 6. Odpojenie zachytávača
handle.remove()
handle_relu.remove()


# 7. Výpis výsledkov
print("Skutočná číslica:", label)
print("Rozmer vstupného obrázka:", image.shape)
print("Rozmer výstupu modelu:", output.shape)
print("Rozmer máp príznakov:", activations["conv1"].shape)
print("Počet máp príznakov:", activations["conv1"].shape[1])
print(
    "Minimum po ReLU:",
    activations_after_relu["relu1"].min().item(),
)

# Kontrola rozdielov medzi mapami príznakov

# Kontrola hodnôt máp po aktivácii ReLU
feature_maps = activations_after_relu["relu1"][0]

for i in range(16):
    feature_map = feature_maps[i]

    print(
        f"Mapa {i + 1}: "
        f"minimum = {feature_map.min().item():.4f}, "
        f"maximum = {feature_map.max().item():.4f}, "
        f"priemer = {feature_map.mean().item():.4f}"
    )



import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 8. Príprava máp príznakov na zobrazenie
feature_maps = activations["conv1"][0]


# 9. Vytvorenie rozloženia 4 x 4
fig = make_subplots(
    rows=4,
    cols=4,
    subplot_titles=[
        f"Filter {i + 1}" for i in range(16)
    ],
    horizontal_spacing=0.04,
    vertical_spacing=0.08,
)


# 10. Pridanie jednotlivých máp do grafu
for i in range(16):
    row = i // 4 + 1
    col = i % 4 + 1

    fig.add_trace(
        
go.Heatmap(
    z=feature_maps[i].numpy(),
    colorscale="RdBu",
    zmin=-2,
    zmax=2,
    showscale=False,
    ),

        row=row,
        col=col,
    )


# 11. Nastavenie vzhľadu grafu
fig.update_layout(
    title="Mapy príznakov prvej konvolučnej vrstvy",
    height=800,
    width=900,
)

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(
    autorange="reversed",
    showticklabels=False,
)

fig.show()