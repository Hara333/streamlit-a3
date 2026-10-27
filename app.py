"""Sentence sentiment prediction (Streamlit).

Run:
    pip install -r requirements.txt
    python train_model.py           # builds model.pkl from the model pages
    streamlit run app.py

bge-large turns the sentence into 2,048 features (loaded in the app, downloaded on first run);
the chosen head from model.pkl turns those into negative / neutral / positive.
"""
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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
    """The [CLS] vector (1,024) then the mean over the tokens (1,024), same as fd_encode_server.py.

    Also returns the tokens and each token's own 1,024-number vector."""
    tok, model = load_encoder()
    enc = tok(sentence, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    mask = enc["attention_mask"][0].unsqueeze(-1).float()
    x = torch.cat([h[0], (h * mask).sum(0) / mask.sum()]).numpy().astype(np.float64)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    return x, tokens, h.numpy().astype(np.float64)


def to_excel(res):
    """One workbook: the prediction, the 2,048-number sentence vector and every token's vector."""
    x, half = res["x"], len(res["x"]) // 2
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame({"sentence": [res["sentence"]], "model": [res["model"]], "label": [res["label"]],
                      **{f"p_{c}": [p] for c, p in zip(CLS, res["probs"])}}
                     ).to_excel(xl, sheet_name="prediction", index=False)
        pd.DataFrame({"position": np.arange(len(x)),
                      "part": ["[CLS] vector"] * half + ["token average"] * (len(x) - half),
                      "value": x}).to_excel(xl, sheet_name="sentence_vector", index=False)
        tv = pd.DataFrame(res["token_vecs"], columns=[f"d{i}" for i in range(res["token_vecs"].shape[1])])
        tv.insert(0, "token", res["tokens"])
        tv.insert(0, "position", np.arange(len(res["tokens"])))
        tv.to_excel(xl, sheet_name="token_vectors", index=False)
    return buf.getvalue()


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
            x, tokens, token_vecs = encode(sentence)
        probs = run_head(heads[model_name], x)
        # keep the result in session state so it survives reruns from other widgets
        st.session_state["prediction"] = {"sentence": sentence, "model": model_name, "probs": probs,
                                          "label": CLS[probs.index(max(probs))], "x": x,
                                          "tokens": tokens, "token_vecs": token_vecs}

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
    st.subheader(f"Sentence vector ({len(x):,} numbers)")
    st.caption(f"bge-large's [CLS] vector (values 0 to {half - 1}) followed by the average of all "
               f"token vectors (values {half} to {len(x) - 1}). The head turns these into the 3 "
               "probabilities above.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Min", f"{x.min():.3f}")
    c2.metric("Mean", f"{x.mean():.3f}")
    c3.metric("Max", f"{x.max():.3f}")
    vfig = go.Figure([go.Bar(x=np.arange(half), y=x[:half], name="[CLS] vector", marker_color="#3b82f6"),
                      go.Bar(x=np.arange(half, len(x)), y=x[half:], name="token average",
                             marker_color="#f59e0b")])
    vfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), bargap=0,
                       xaxis_title="position in vector", yaxis_title="value",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(vfig, width="stretch")
    with st.expander("All 2,048 values"):
        st.dataframe({"position": np.arange(len(x)),
                      "part": ["[CLS] vector"] * half + ["token average"] * (len(x) - half),
                      "value": np.round(x, 5)}, hide_index=True, width="stretch")

    # ---------- one 1,024-number vector per token ----------
    tokens, tv = res["tokens"], res["token_vecs"]
    st.subheader(f"Token vectors ({len(tokens)} tokens × {tv.shape[1]:,} numbers)")
    st.caption("bge-large gives every token its own vector. Words it doesn't know whole are split into "
               "pieces (a piece starting with ## continues the word before it). [CLS] and [SEP] mark the "
               "start and end; the [CLS] row is the first half of the sentence vector, and the average of "
               "all rows is the second half.")
    labels = [f"{i}: {t}" for i, t in enumerate(tokens)]
    lim = float(np.abs(tv).max())
    hfig = go.Figure(go.Heatmap(z=tv, y=labels, colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
                                colorbar=dict(title="value"),
                                hovertemplate="%{y}<br>position %{x}<br>value %{z:.4f}<extra></extra>"))
    hfig.update_layout(height=max(220, 26 * len(tokens) + 60), margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="position in token vector", yaxis=dict(autorange="reversed"))
    st.plotly_chart(hfig, width="stretch")

    pick = st.selectbox("Look at one token", labels)
    row = tv[labels.index(pick)]
    tfig = go.Figure(go.Bar(x=np.arange(len(row)), y=row, marker_color="#3b82f6"))
    tfig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), bargap=0,
                       xaxis_title="position in token vector", yaxis_title="value")
    st.plotly_chart(tfig, width="stretch")

    st.download_button("Download as Excel", to_excel(res), file_name="sentence_vectors.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:")
