import streamlit as st
import torch
import torch.nn.functional as F
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
from torchvision import datasets, transforms

st.set_page_config(
    page_title="CNN Lab | Konvolúcia",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 CNN Lab")
st.subheader("Interaktívna konvolúcia")
st.write(
    "Sleduj, ako filter postupuje po obrázku, "
    "násobí hodnoty a vytvára výstupnú feature mapu."
)


# Vstupná matica 5 × 5

from torchvision import datasets, transforms

# Načítanie testovacej množiny MNIST
mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

# Vyberieme jednu skutočnú číslicu
image_tensor, label = mnist[0]

# Pre vizualizáciu prevedieme obrázok na maticu 28 × 28
image = image_tensor[0]
image = image.to(torch.float32)

st.subheader("Vstupný obrázok MNIST")

st.image(
    image_tensor[0].numpy(),
    caption=f"Skutočná číslica: {label}",
    clamp=True,
    width=280,
)

# Načítanie skutočných naučených filtrov CNN
from src.model import MNISTCNN

model = MNISTCNN()
state_dict = torch.load(
    "model.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

# Váhy prvej konvolučnej vrstvy: 16 filtrov, každý 3 × 3
learned_filters = model.features[0].weight.detach().cpu()

# Vyberieme prvý naučený filter
filter_index = st.selectbox(
    "Vyber naučený filter:",
    options=list(range(16)),
    format_func=lambda i: f"Filter {i + 1}",
)
kernel = learned_filters[filter_index, 0].clone()


# Skutočný výpočet prvej konvolučnej vrstvy natrénovanej CNN
conv1 = model.features[0]

image_4d = image_tensor.unsqueeze(0)

feature_maps = conv1(image_4d)[0]
feature_map = feature_maps[filter_index]

# Naučený filter a jeho bias
kernel = conv1.weight[filter_index, 0].detach().cpu().clone()
conv_bias = conv1.bias[filter_index].item()

# Vybraná bunka vo feature mape
if "selected_row" not in st.session_state:
    st.session_state.selected_row = 0

if "selected_col" not in st.session_state:
    st.session_state.selected_col = 0


output_size = feature_map.shape[0]

st.subheader("Vyber bunku na analýzu")

col_row, col_col = st.columns(2)

with col_row:
    selected_row = st.number_input(
        "Riadok",
        min_value=0,
        max_value=output_size - 1,
        value=st.session_state.selected_row,
        step=1,
    )

with col_col:
    selected_col = st.number_input(
        "Stĺpec",
        min_value=0,
        max_value=output_size - 1,
        value=st.session_state.selected_col,
        step=1,
    )

st.session_state.selected_row = int(selected_row)
st.session_state.selected_col = int(selected_col)
total_steps = feature_map.numel()




# --- Interaktívna vizualizácia konvolúcie ---

import numpy as np

image_data = image.numpy()
kernel_data = kernel.numpy()
result_data = feature_map.detach().cpu().numpy()

output_height, output_width = result_data.shape

# Ochrana pred výberom bunky mimo rozsahu
selected_row = min(
    max(0, st.session_state.selected_row),
    output_height - 1,
)
selected_col = min(
    max(0, st.session_state.selected_col),
    output_width - 1,
)

# Výpočet okna pre vybranú bunku
window = image[
    selected_row:selected_row + 3,
    selected_col:selected_col + 3,
]
products = window * kernel

# Súčet násobení a naučený bias
calculated_value = products.sum().item() + conv_bias

st.markdown("### Interaktívna konvolúcia")
st.caption(
    "Klikni na ľubovoľnú bunku vo feature mape "
    "a zobrazí sa jej výpočet."
)

col_image, col_filter, col_feature = st.columns(
    [1.2, 0.65, 1.5]
)

# 1. Vstupná matica s vyznačeným oknom 3 × 3
with col_image:
    st.markdown("#### Vstup")

    input_fig = go.Figure(
        go.Heatmap(
            z=image_data,
            colorscale="Greys",
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=(
                "Riadok: %{y}<br>"
                "Stĺpec: %{x}<br>"
                "Hodnota: %{z:.0f}<extra></extra>"
            ),
        )
    )

    input_fig.add_shape(
        type="rect",
        x0=selected_col - 0.5,
        x1=selected_col + 2.5,
        y0=selected_row - 0.5,
        y1=selected_row + 2.5,
        line=dict(color="#F59E0B", width=3),
        fillcolor="rgba(245,158,11,0.20)",
    )

    input_fig.update_yaxes(autorange="reversed")
    input_fig.update_layout(
        height=340,
        margin=dict(l=5, r=5, t=5, b=5),
    )
    st.plotly_chart(
        input_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

# 2. Malý konvolučný filter
with col_filter:
    st.markdown("#### Filter")

    filter_fig = go.Figure(
        go.Heatmap(
            z=kernel_data,
            x=[0, 1, 2],
            y=[0, 1, 2],
            colorscale=[
                [0.0, "#DC6262"],
                [0.499, "#DC6262"],
                [0.5, "#E5E7EB"],
                [0.501, "#E5E7EB"],
                [1.0, "#269B72"],
            ],
            zmin=-1,
            zmax=1,
            showscale=False,
            text=kernel_data.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=22, color="#111827"),
            xgap=4,
            ygap=4,
            hovertemplate="Váha: %{z}<extra></extra>",
        )
    )

    filter_fig.update_yaxes(autorange="reversed")
    filter_fig.update_layout(
        height=180,
        margin=dict(l=2, r=2, t=5, b=5),
    )
    st.plotly_chart(
        filter_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption("3 × 3 váhy")

# 3. Klikateľná feature mapa
with col_feature:
    st.markdown("#### Feature mapa")

    feature_fig = go.Figure(
        go.Heatmap(
            z=result_data,
            x=list(range(output_width)),
            y=list(range(output_height)),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(thickness=10, len=0.8),
            xgap=1,
            ygap=1,
            hovertemplate=(
                "Riadok: %{y}<br>"
                "Stĺpec: %{x}<br>"
                "Výsledok: %{z:.1f}<extra></extra>"
            ),
        )
    )

    feature_fig.add_shape(
        type="rect",
        x0=selected_col - 0.5,
        x1=selected_col + 0.5,
        y0=selected_row - 0.5,
        y1=selected_row + 0.5,
        line=dict(color="#F59E0B", width=3),
        fillcolor="rgba(245,158,11,0.25)",
    )

    feature_fig.update_yaxes(autorange="reversed")
    feature_fig.update_layout(
        height=340,
        margin=dict(l=5, r=5, t=5, b=5),
    )

    st.plotly_chart(
        feature_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

# 4. Detail výpočtu vybratej bunky
st.divider()
st.markdown("### Výpočet vybratej bunky")

m1, m2, m3 = st.columns(3)
m1.metric(
    "Riadok",
    selected_row + 1,
)
m2.metric(
    "Stĺpec",
    selected_col + 1,
)
m3.metric(
    "Výsledok konvolúcie",
    f"{calculated_value:.2f}",
)

st.markdown("#### Jednotlivé násobenia")

detail_cols = st.columns(3)

for r in range(3):
    for c in range(3):
        a = window[r, c].item()
        b = kernel[r, c].item()
        product = products[r, c].item()

        with detail_cols[c]:
            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding:12px 4px;
                    border:1px solid #64748b;
                    border-radius:10px;
                    margin-bottom:8px;
                ">
                    <div style="font-size:12px;color:#94a3b8">
                        {a:g} × {b:g}
                    </div>
                    <div style="font-size:20px;font-weight:600">
                        {product:g}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("#### Súčet násobení a bias")

terms = []

for r in range(3):
    for c in range(3):
        terms.append(
            f"({window[r, c].item():.4f} × {kernel[r, c].item():.4f})"
        )

st.code(
    " + ".join(terms)
    + f"\n+ bias ({conv_bias:.4f})"
    + f"\n\n= {calculated_value:.4f}",
    language="text",
)