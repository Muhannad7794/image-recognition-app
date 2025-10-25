# dsp-server.py
import os, json, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
import numpy as np

print("[BOOT] starting server module import", flush=True)

from dsp import generate_features 

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "parameters.json")

def _read_parameters_json():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"parameters.json not found at {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Studio expects a top-level version field in /parameters
    data["version"] = 1
    return data

def _send_json(handler, code, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(body)

def _single_request(handler, body):
    if "features" not in body:       raise ValueError('Missing "features"')
    if "params" not in body:         raise ValueError('Missing "params"')
    if "sampling_freq" not in body:  raise ValueError('Missing "sampling_freq"')
    if "draw_graphs" not in body:    body["draw_graphs"] = False

    args = {
        "draw_graphs": body["draw_graphs"],
        "raw_data": np.array(body["features"]),
        "axes": body.get("axes", []),
        "sampling_freq": body["sampling_freq"],
        "implementation_version": body.get("implementation_version", 1),
    }
    for k, v in body["params"].items():
        args[k] = v

    res = generate_features(**args)
    if isinstance(res.get("features"), np.ndarray):
        res["features"] = res["features"].tolist()
    _send_json(handler, 200, res)

def _batch_request(handler, body):
    if "features" not in body:       raise ValueError('Missing "features"')
    if "params" not in body:         raise ValueError('Missing "params"')
    if "sampling_freq" not in body:  raise ValueError('Missing "sampling_freq"')

    base_args = {
        "draw_graphs": False,
        "axes": body.get("axes", []),
        "sampling_freq": body["sampling_freq"],
        "implementation_version": body.get("implementation_version", 1),
    }
    for k, v in body["params"].items():
        base_args[k] = v

    features_out, labels, output_config = [], [], None
    for idx, example in enumerate(body["features"]):
        args = dict(base_args)
        args["raw_data"] = np.array(example)
        res = generate_features(**args)
        f = res.get("features")
        if isinstance(f, np.ndarray): f = f.tolist()
        features_out.append(f)
        if idx == 0:
            labels = res.get("labels", [])
            output_config = res.get("output_config")
    _send_json(handler, 200, {
        "success": True,
        "features": features_out,
        "labels": labels,
        "output_config": output_config
    })

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        print(f"[GET] {url.path}", flush=True)
        try:
            if url.path == "/":
                self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
                self.wfile.write(b"Edge Impulse DSP block server\n"); return
            if url.path == "/parameters":
                params = _read_parameters_json()
                _send_json(self, 200, params); return
            self.send_error(404, "Invalid path")
        except Exception as e:
            traceback.print_exc()
            _send_json(self, 200, {"success": False, "error": str(e)})

    def do_POST(self):
        url = urlparse(self.path)
        print(f"[POST] {url.path}", flush=True)
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_len).decode("utf-8") or "{}")
            if url.path == "/run":   _single_request(self, body); return
            if url.path == "/batch": _batch_request(self, body); return
            self.send_error(404, "Invalid path")
        except Exception as e:
            traceback.print_exc()
            _send_json(self, 200, {"success": False, "error": str(e)})

    def log_message(self, fmt, *args):  # quiet default logs
        return

class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def run():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "80"))  # default 80 now
    print(f"Listening on host {host} port {port}", flush=True)
    server = ThreadingSimpleServer((host, port), Handler)
    server.serve_forever()

if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
