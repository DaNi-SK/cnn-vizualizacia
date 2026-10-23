import streamlit as st
import torch
from torchvision import datasets, transforms
from src.model import MNISTCNN
from streamlit_plotly_events import plotly_events
from training_utils import train_model

st.set_page_config(
    page_title="CNN Lab | AI Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bočný panel
with st.sidebar:
    st.selectbox("Choose a model", ["GPT-4", "Claude", "Gemini"])
    st.divider()


if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

if "training_history" not in st.session_state:
    st.session_state["training_history"] = None

model = st.session_state["trained_model"]

# Načítanie testovacích obrázkov MNIST
@st.cache_data
def load_mnist():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    return dataset


mnist = load_mnist()


# Výpočet konvolúcie nad obrázkom
def calculate_convolution(image, kernel):
    image = image.squeeze()

    # Obrázok doplníme nulami, aby sa zachovala jeho veľkosť
    padded_image = torch.nn.functional.pad(
        image.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1),
    ).squeeze()

    # Výpočet konvolúcie pomocou PyTorch
    result = torch.nn.functional.conv2d(
        padded_image.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
    )

    return result.squeeze()


def create_convolution_visual(image, activation, row_pos, col_pos):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    image_array = image.squeeze().numpy()
    activation_array = activation.detach().numpy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Vstupná číslica — okno filtra 3 × 3",
            "Aktivačná mapa — zodpovedajúca pozícia",
        ),
        horizontal_spacing=0.12,
    )

    
    
    fig.add_trace(
        go.Heatmap(
            z=image_array,
            x=list(range(image_array.shape[1])),
            y=list(range(image_array.shape[0])),
            colorscale=[
                [0, "#050816"],
                [1, "#F2F5FF"],
            ],
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=(
                "Riadok: %{y}<br>"
                "Stĺpec: %{x}<br>"
                "Pixel: %{z:.3f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )



    fig.add_trace(
        go.Heatmap(
            z=activation_array,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Aktivácia", x=1.02),
            hovertemplate="Riadok: %{y}<br>Stĺpec: %{x}<br>Hodnota: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Červený rám označuje aktuálne okno 3 × 3.
    fig.add_shape(
        type="rect",
        x0=col_pos - 0.5,
        x1=col_pos + 2.5,
        y0=row_pos - 0.5,
        y1=row_pos + 2.5,
        line=dict(color="#ff4d6d", width=3),
        fillcolor="rgba(255,77,109,0.15)",
        row=1,
        col=1,
    )

    # Stred okna zodpovedá rovnakej pozícii v aktivačnej mape.
    fig.add_trace(
        go.Scatter(
            x=[col_pos + 1],
            y=[row_pos + 1],
            mode="markers",
            marker=dict(
                symbol="square-open",
                size=20,
                color="#ff4d6d",
                line=dict(width=3),
            ),
            name="Výsledná pozícia",
            hovertemplate="Výstupný riadok: %{y}<br>Stĺpec: %{x}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)


    fig.update_layout(
        height=480,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=65, t=65, b=10),
        font=dict(color="#dce5f7"),
        showlegend=False,
    )

    return fig


def capture_layer_outputs(model, image):
    """Zaznamená skutočné výstupy všetkých vrstiev CNN."""
    model.eval()

    outputs = {}
    x = image.unsqueeze(0)

    with torch.no_grad():
        for index, layer in enumerate(model.features):
            x = layer(x)
            outputs[f"features.{index} — {layer.__class__.__name__}"] = (
                x.detach().cpu().clone()
            )

        for index, layer in enumerate(model.classifier):
            x = layer(x)
            outputs[f"classifier.{index} — {layer.__class__.__name__}"] = (
                x.detach().cpu().clone()
            )

    return outputs

 # Úvodná stránka — 11 krokov výpočtu CNN

steps = [
    "01 · Vstupný obrázok",
    "02 · Konvolúcia 1",
    "03 · ReLU 1",
    "04 · Max Pooling 1",
    "05 · Konvolúcia 2",
    "06 · ReLU 2",
    "07 · Max Pooling 2",
    "08 · Flatten",
    "09 · Plne prepojená vrstva",
    "10 · ReLU 3",
    "11 · Výstupná vrstva",
]

step_descriptions = [
    "Vstupom siete je obrázok číslice z datasetu MNIST s rozmermi 28 × 28 pixelov.",
    "Konvolučný filter 3 × 3 prechádza cez obrázok a vytvára mapy príznakov.",
    "ReLU nahrádza záporné hodnoty nulou a ponecháva kladné hodnoty.",
    "Max Pooling vyberá maximum z oblasti 2 × 2 a zmenšuje priestorové rozmery.",
    "Druhá konvolučná vrstva spracúva príznaky získané z predchádzajúcej vrstvy.",
    "Druhá funkcia ReLU aplikuje nelinearitu na výstup druhej konvolúcie.",
    "Druhá vrstva Max Pooling opäť zmenšuje rozmery aktivačných máp.",
    "Flatten prevedie viacrozmerné mapy príznakov na jeden vektor.",
    "Lineárna vrstva spracuje vektor a vypočíta 128 skrytých hodnôt.",
    "ReLU aplikuje nelinearitu na výstup skrytej lineárnej vrstvy.",
    "Výstupná vrstva vytvorí 10 skóre pre číslice od 0 do 9.",
]

if "selected_step" not in st.session_state:
    st.session_state["selected_step"] = 0

st.markdown("""
<div class="hero">
    <div class="eyebrow">CNN · FORWARD PASS</div>
    <div class="hero-title">
        Priebeh výpočtu <span class="gradient-text">neurónovej siete</span>
    </div>
    <div class="subtitle">
        Vyber jednotlivé kroky a preskúmaj, ako CNN spracúva obrázok číslice.
    </div>
</div>
""", unsafe_allow_html=True)



# Kroky výpočtu CNN

forward_steps = [
    "Vstup",
    "Conv 1",
    "ReLU 1",
    "Pool 1",
    "Conv 2",
    "ReLU 2",
    "Pool 2",
    "Flatten",
    "Linear",
    "ReLU 3",
    "Výstup",
]

backward_steps = [
    "Výstupný gradient",
    "Linear gradient",
    "ReLU gradient",
    "Unflatten",
    "Pool 2 gradient",
    "Conv 2 gradient",
    "Pool 1 gradient",
    "Conv 1 gradient",
    "Gradienty váh",
    "Aktualizácia váh",
]

if "selected_step" not in st.session_state:
    st.session_state["selected_step"] = 0

if "selected_phase" not in st.session_state:
    st.session_state["selected_phase"] = "forward"


def show_steps(labels, phase, reverse=False):
    # Backpropagation zobrazíme v opačnom poradí.
    displayed_steps = list(enumerate(labels))

    if reverse:
        displayed_steps.reverse()

    # Medzi krokmi sa striedajú tlačidlá a vodorovné šípky.
    columns = st.columns(len(displayed_steps) * 2 - 1)

    for position, (index, label) in enumerate(displayed_steps):
        with columns[position * 2]:
            selected = (
                st.session_state["selected_phase"] == phase
                and st.session_state["selected_step"] == index
            )

            st.button(
                f"{index + 1}. {label}",
                key=f"{phase}_button_{index}",
                type="primary" if selected else "secondary",
                use_container_width=True,
                on_click=lambda i=index, p=phase: (
                    st.session_state.update(
                        selected_step=i,
                        selected_phase=p,
                    )
                ),
            )

        if position < len(displayed_steps) - 1:
            with columns[position * 2 + 1]:
                arrow = "←" if reverse else "→"
                st.markdown(
                    f"<div style='text-align:center; "
                    f"padding-top:12px; font-size:22px;'>"
                    f"{arrow}</div>",
                    unsafe_allow_html=True,
                )



# Poradie krokov pri prechádzaní aplikáciou
all_steps = (
    [("forward", i) for i in range(len(forward_steps))]
    + [("backward", i) for i in reversed(range(len(backward_steps)))]
)


def move_step(direction):
    current = (
        st.session_state["selected_phase"],
        st.session_state["selected_step"],
    )

    position = all_steps.index(current)
    position = max(0, min(len(all_steps) - 1, position + direction))

    phase, index = all_steps[position]
    st.session_state["selected_phase"] = phase
    st.session_state["selected_step"] = index


st.title("Vizualizácia CNN")

st.subheader("1. Forward pass — výpočet dopredu")
show_steps(forward_steps, "forward")

st.divider()

st.subheader("2. Backpropagation — spätné šírenie chyby")
show_steps(backward_steps, "backward", reverse=True)

st.divider()

# Aktuálne vybraný krok
phase = st.session_state["selected_phase"]
selected = st.session_state["selected_step"]

if phase == "forward":
    st.subheader(forward_steps[selected])
    st.write(
        "Tu bude vysvetlenie a vizualizácia vybraného kroku "
        "dopredného výpočtu."
    )
else:
    st.subheader(backward_steps[selected])
    st.write(
        "Tu bude vysvetlenie a vizualizácia vybraného kroku "
        "spätného šírenia chyby."
    )

# Navigácia pod vysvetlením
current_position = all_steps.index((phase, selected))
total_steps = len(all_steps)

st.divider()

left, center, right = st.columns([1, 2, 1])

with left:
    st.button(
        "← Predchádzajúci",
        key="previous_step",
        use_container_width=True,
        disabled=(current_position == 0),
        on_click=move_step,
        args=(-1,),
    )

with center:
    st.markdown(
        f"<div style='text-align:center; padding-top:8px; "
        f"font-size:20px; font-weight:bold;'>"
        f"{current_position + 1} / {total_steps}</div>",
        unsafe_allow_html=True,
    )

with right:
    st.button(
        "Ďalší →",
        key="next_step",
        use_container_width=True,
        disabled=(current_position == total_steps - 1),
        on_click=move_step,
        args=(1,),
    )

st.stop()

