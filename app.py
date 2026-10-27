"""Sentence sentiment prediction (Streamlit).

Run:
    pip install -r requirements.txt
    python train_model.py           # builds model.pkl from the model pages
    streamlit run app.py

bge-large turns the sentence into 2,048 features (loaded in the app, downloaded on first run);
the chosen head from model.pkl turns those into negative / neutral / positive.
"""
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

st.set_page_config(page_title="Sentiment prediction", page_icon="💬")
HERE = Path(__file__).resolve().parent
ENC_NAME = "BAAI/bge-large-en-v1.5"

CLS = ["negative", "neutral", "positive"]
CLS_COL = ["#e5484d", "#8b5cf6", "#1f9d55"]
ICON = {"negative": "🔴", "neutral": "🟣", "positive": "🟢"}


# ---------- head ----------
ACT = {
    "relu": lambda x: np.maximum(x, 0),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "silu": lambda x: x / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "elu": lambda x: np.where(x > 0, x, np.exp(x) - 1),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}


@st.cache_resource
def load_heads():
    """Load the heads saved by train_model.py."""
    with open(HERE / "model.pkl", "rb") as f:
        return pickle.load(f)


def run_head(h, x):
    a = (x - h["mean"]) / h["sd"]
    for W, b, act in h["trunk"]:
        a = ACT[act](W @ a + b)
    W, b = h["out"]
    z = (W @ a + b) / h["T"]
    p = np.exp(z - z.max())
    p /= p.sum()
    by_label = dict(zip(h["classes"], p))
    return [float(by_label[c]) for c in CLS]


@st.cache_resource
def load_encoder():
    return AutoTokenizer.from_pretrained(ENC_NAME), AutoModel.from_pretrained(ENC_NAME).eval()


def encode(sentence):
    """The [CLS] vector (1,024) then the mean over the tokens (1,024), same as fd_encode_server.py."""
    tok, model = load_encoder()
    enc = tok(sentence, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    mask = enc["attention_mask"][0].unsqueeze(-1).float()
    return torch.cat([h[0], (h * mask).sum(0) / mask.sum()]).numpy().astype(np.float64)


# ---------- page ----------
st.title("Sentiment prediction")
if not (HERE / "model.pkl").exists():
    st.error("model.pkl not found. Run `python train_model.py` first.")
    st.stop()
heads = load_heads()

sentence = st.text_area("Sentence", placeholder="e.g. Shares jumped after the company beat earnings forecasts.",
                        height=100)
model_name = st.selectbox("Model", list(heads))

if st.button("Predict", type="primary"):
    if not sentence.strip():
        st.warning("Type a sentence first.")
    else:
        with st.spinner("Encoding with bge-large (the first run downloads it, about 1.3 GB) ..."):
            x = encode(sentence)
        probs = run_head(heads[model_name], x)
        # keep the result in session state so it survives reruns from other widgets
        st.session_state["prediction"] = {"sentence": sentence, "model": model_name, "probs": probs,
                                          "label": CLS[probs.index(max(probs))], "x": x}

res = st.session_state.get("prediction")
if res:
    st.subheader("Result")
    if res["sentence"] != sentence or res["model"] != model_name:
        st.caption(f"Result for: “{res['sentence']}” with {res['model']}")
    st.metric("Sentiment", f"{ICON[res['label']]} {res['label']}",
              f"{max(res['probs']):.0%} confident", delta_color="off")
    fig = go.Figure(go.Bar(x=res["probs"], y=CLS, orientation="h", marker_color=CLS_COL,
                           text=[f"{v:.0%}" for v in res["probs"]], textposition="outside"))
    fig.update_layout(height=200, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(range=[0, 1.1], tickformat=".0%"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    # ---------- the 2,048 features the head read ----------
    x = res["x"]
    half = len(x) // 2
    with st.expander("All 2,048 values"):
        st.dataframe({"position": np.arange(len(x)),
                      "part": ["[CLS] vector"] * half + ["token average"] * (len(x) - half),
                      "value": np.round(x, 5)}, hide_index=True, width="stretch")
