"""Local bge-large encoder for the sentiment heads.

Run:
    python fd_encode_server.py

Serves http://127.0.0.1:8765
    GET /health               -> {"ok": true}
    GET /encode?text=...      -> {"values": [2,048 floats], "text", "seconds", "tokens"}

The 2,048 values are bge-large-en-v1.5's last-layer [CLS] vector (1,024), then the mean over
its tokens (1,024), both unnormalised: the features the heads were trained on.
"""
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModel, AutoTokenizer

HOST, PORT = "127.0.0.1", 8765
NAME = "BAAI/bge-large-en-v1.5"

print(f"loading {NAME} ...")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).eval()


def encode(text):
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    mask = enc["attention_mask"][0].unsqueeze(-1).float()
    return torch.cat([h[0], (h * mask).sum(0) / mask.sum()]).tolist(), int(mask.sum())


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # the model pages call this from the browser
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/health":
            return self.send_json(200, {"ok": True})
        if url.path == "/encode":
            text = urllib.parse.parse_qs(url.query).get("text", [""])[0].strip()
            if not text:
                return self.send_json(400, {"error": "empty text"})
            t0 = time.time()
            values, tokens = encode(text)
            return self.send_json(200, {"values": values, "text": text,
                                        "seconds": round(time.time() - t0, 2), "tokens": tokens})
        self.send_json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"encoder ready at http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
