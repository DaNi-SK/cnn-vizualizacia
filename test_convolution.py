
import torch
import torch.nn.functional as F


# 1. Jednoduchý vstupný obrázok 5 x 5
image = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
], dtype=torch.float32)


# 2. Filter 3 x 3
kernel = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=torch.float32)


# 3. Príprava rozmerov pre PyTorch
image = image.unsqueeze(0).unsqueeze(0)
kernel = kernel.unsqueeze(0).unsqueeze(0)


# 4. Výpočet konvolúcie bez paddingu
feature_map = F.conv2d(image, kernel)


# 5. Výpis výsledkov
print("Rozmer obrázka:", image.shape)
print("Rozmer filtra:", kernel.shape)
print("Výsledná feature mapa:")
print(feature_map[0, 0])
print("Rozmer feature mapy:", feature_map.shape)


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Prevod PyTorch tenzorov na dvojrozmerné dáta
image_2d = image[0, 0].numpy()
kernel_2d = kernel[0, 0].numpy()
feature_map_2d = feature_map[0, 0].numpy()


# Vytvorenie troch grafov vedľa seba
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "Vstupný obrázok (5 × 5)",
        "Konvolučný filter (3 × 3)",
        "Feature mapa (3 × 3)",
    ),
    horizontal_spacing=0.12,
)


# 1. Vstupný obrázok
fig.add_trace(
    go.Heatmap(
        z=image_2d,
        colorscale="Blues",
        text=image_2d,
        texttemplate="%{text}",
        showscale=False,
    ),
    row=1,
    col=1,
)


# 2. Konvolučný filter
fig.add_trace(
    go.Heatmap(
        z=kernel_2d,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=kernel_2d,
        texttemplate="%{text}",
        showscale=False,
    ),
    row=1,
    col=2,
)


# 3. Výsledná feature mapa
fig.add_trace(
    go.Heatmap(
        z=feature_map_2d,
        colorscale="RdBu",
        text=feature_map_2d,
        texttemplate="%{text}",
        showscale=False,
    ),
    row=1,
    col=3,
)


# Nastavenie osí a rozloženia
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    title="Vizualizácia konvolúcie",
    height=450,
    width=1100,
)

fig.show()
