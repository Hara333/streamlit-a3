"""Sentence sentiment prediction (Streamlit).

Run:
    pip install -r requirements.txt
    python train_model.py           # builds model.pkl from the model pages
    python fd_encode_server.py      # the bge-large encoder at 127.0.0.1:8765
    streamlit run app.py

The encoder turns the sentence into 2,048 features; the chosen head from model.pkl turns
those into negative / neutral / positive.
"""
import json
import pickle
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sentiment prediction", page_icon="💬")
HERE = Path(__file__).resolve().parent
ENC_URL = "http://127.0.0.1:8765"

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


def encode(sentence):
    url = f"{ENC_URL}/encode?text={urllib.parse.quote(sentence)}"
    with urllib.request.urlopen(url, timeout=90) as r:
        return np.array(json.loads(r.read())["values"], dtype=np.float64)


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
        try:
            with st.spinner("Encoding with bge-large ..."):
                x = encode(sentence)
        except OSError:
            st.error(f"Encoder offline. Start fd_encode_server.py so it answers at {ENC_URL}.")
        else:
            probs = run_head(heads[model_name], x)
            # keep the result in session state so it survives reruns from other widgets
            st.session_state["prediction"] = {"sentence": sentence, "model": model_name, "probs": probs,
                                              "label": CLS[probs.index(max(probs))]}

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
