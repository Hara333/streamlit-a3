"""Build model.pkl for app.py.

Run:
    python train_model.py

Each head's trained weights are read from its model page (model_a.html, model_a_sentfin.html,
dialogue_head_fingpt_sentfin.html) in this folder and saved together in model.pkl, so app.py
only has to load one file.
"""
import base64
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

MODELS = {
    "Model A": "model_a.html",
    "Model A + SEntFiN": "model_a_sentfin.html",
    "Dialogue + FinGPT + SEntFiN": "dialogue_head_fingpt_sentfin.html",
}


def dec_f(b):
    return np.frombuffer(base64.b64decode(b), dtype="<f4").astype(np.float64)


def load_head(file):
    """Unpack the head stored in a model page: input scaling, hidden layers and the sentiment output."""
    s = (HERE / file).read_text(encoding="utf-8")
    start = s.index("const D = ") + len("const D = ")
    m = json.loads(s[start:s.index("</script>", start)].strip().rstrip(";"))["model"]
    layer = lambda l: (dec_f(l["W"]).reshape(l["rows"], l["cols"]), np.array(l["b"]))
    out = next(o for o in m["outs"] if o["reading"] == "sentiment")
    return {"mean": dec_f(m["mean"]), "sd": dec_f(m["sd"]),
            "trunk": [(*layer(l), l["act"]) for l in m["trunk"]],
            "out": layer(out), "T": out["temperature"], "classes": out["classes"],
            "arch": m["arch"]}


if __name__ == "__main__":
    heads = {}
    for name, file in MODELS.items():
        heads[name] = load_head(file)
        print(f"{name}: {heads[name]['arch']}")

    with open(HERE / "model.pkl", "wb") as f:
        pickle.dump(heads, f)

    print("Model saved as model.pkl")
