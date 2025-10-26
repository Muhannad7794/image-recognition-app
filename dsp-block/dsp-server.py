# dsp-server.py — resilient multi-port server for Edge Impulse
import os, json, traceback, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
import numpy as np

print("[BOOT] starting server module import", flush=True)
try:
    from dsp import generate_features
    print("[BOOT] dsp.py imported OK", flush=True)
except Exception:
    traceback.print_exc()
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "parameters.json")

def _read_parameters_json():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"parameters.json not found at {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # EI expects a top-level "version"
    if "version" not in data:
        data["version"] = 1
    return data

def _send_json(h, code, payload):
    body = json.dumps(payload).encode("utf-8")
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    try: h.wfile.write(body)
    except BrokenPipeError: pass

def _send_plain(h, code, text="OK"):
    body = text.encode("utf-8")
    h.send_response(code)
    h.send_header("Content-Type", "text/plain")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    try: h.wfile.write(body)
    except BrokenPipeError: pass

def _single_request(h, body):
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
    _send_json(h, 200, res)

def _batch_request(h, body):
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
    feats, labels, output_config = [], [], None
    for idx, ex in enumerate(body["features"]):
        args = dict(base_args); args["raw_data"] = np.array(ex)
        res = generate_features(**args)
        f = res.get("features")
        if isinstance(f, np.ndarray): f = f.tolist()
        feats.append(f)
        if idx == 0:
            labels = res.get("labels", [])
            output_config = res.get("output_config")
    _send_json(h, 200, {"success": True, "features": feats,
                        "labels": labels, "output_config": output_config})

class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        p = urlparse(self.path).path
        print(f"[HEAD] {p}", flush=True)
        try:
            if p in ("/", "/health"):
                self.send_response(200); self.end_headers(); return
            if p == "/parameters":
                _ = _read_parameters_json()
                self.send_response(200); self.end_headers(); return
            self.send_error(404, "Invalid path")
        except Exception as e:
            traceback.print_exc()
            try: _send_json(self, 200, {"success": False, "error": str(e)})
            except Exception: pass

    def do_GET(self):
        p = urlparse(self.path).path
        print(f"[GET ] {p}", flush=True)
        try:
            if p in ("/", "/health"):
                _send_plain(self, 200, "OK"); return
            if p == "/parameters":
                _send_json(self, 200, _read_parameters_json()); return
            self.send_error(404, "Invalid path")
        except Exception as e:
            traceback.print_exc()
            _send_json(self, 200, {"success": False, "error": str(e)})

    def do_POST(self):
        p = urlparse(self.path).path
        print(f"[POST] {p}", flush=True)
        try:
            ln = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(ln) if ln > 0 else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            if p == "/run":   _single_request(self, body); return
            if p == "/batch": _batch_request(self, body); return
            self.send_error(404, "Invalid path")
        except Exception as e:
            traceback.print_exc()
            _send_json(self, 200, {"success": False, "error": str(e)})

    def log_message(self, *a): return

class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def _serve(host: str, port: int, started_event: threading.Event):
    try:
        print(f"[BOOT] binding HTTP server on {host}:{port}", flush=True)
        srv = ThreadingSimpleServer((host, port), Handler)
        print(f"Listening on host {host} port {port}", flush=True)
        started_event.set()
        srv.serve_forever()
    except Exception as e:
        # Do NOT crash the process if one port fails; just log it.
        print(f"[BOOT] FAILED to bind {host}:{port} -> {e}", flush=True)
        traceback.print_exc()

def run():
    host = os.environ.get("HOST", "0.0.0.0")
    # Accept PORTS="80,4446" or PORT="4446,80" or single number
    ports_env = os.environ.get("PORTS") or os.environ.get("PORT") or "80,4446"
    ports = []
    for token in str(ports_env).split(","):
        token = token.strip()
        if token:
            try: ports.append(int(token))
            except: pass
    if not ports:
        ports = [4446]  # safe default

    started_any = threading.Event()
    threads = []
    for pt in ports:
        evt = started_any  # reuse the same event; any successful bind sets it
        t = threading.Thread(target=_serve, args=(host, pt, evt), daemon=True)
        t.start()
        threads.append(t)

    # Wait a short grace period for at least one port to bind
    if not started_any.wait(timeout=5.0):
        print("[BOOT] ERROR: No ports could be bound. Exiting.", flush=True)
        raise SystemExit(1)

    # Keep main thread alive
    for t in threads:
        t.join()

if __name__ == "__main__":
    run()
