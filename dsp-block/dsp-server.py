# dsp-server.py
# Generic Edge Impulse DSP server (Python).
# You usually don't need to modify this file.

import os, json, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
import numpy as np

from dsp import generate_features  # <-- your function in dsp.py


def _read_parameters_json():
    with open("parameters.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _single_request(handler, body):
    # Basic validation
    if not body.get("features"):
        raise ValueError('Missing "features" in body')
    if "params" not in body:
        raise ValueError('Missing "params" in body')
    if "sampling_freq" not in body:
        raise ValueError('Missing "sampling_freq" in body')
    if "draw_graphs" not in body:
        raise ValueError('Missing "draw_graphs" in body')

    args = {
        "draw_graphs": body["draw_graphs"],
        "raw_data": np.array(body["features"]),
        "axes": body.get("axes", []),
        "sampling_freq": body["sampling_freq"],
        "implementation_version": body.get("implementation_version", 1),
    }

    # Add params from parameters.json UI
    for k, v in body["params"].items():
        args[k] = v

    result = generate_features(**args)
    if isinstance(result.get("features"), np.ndarray):
        result["features"] = result["features"].tolist()

    _send_json(handler, 200, result)


def _batch_request(handler, body):
    if not body.get("features"):
        raise ValueError('Missing "features" in body')
    if "params" not in body:
        raise ValueError('Missing "params" in body')
    if "sampling_freq" not in body:
        raise ValueError('Missing "sampling_freq" in body')

    base_args = {
        "draw_graphs": False,
        "axes": body.get("axes", []),
        "sampling_freq": body["sampling_freq"],
        "implementation_version": body.get("implementation_version", 1),
    }
    for k, v in body["params"].items():
        base_args[k] = v

    features_out = []
    labels = []
    output_config = None

    for idx, example in enumerate(body["features"]):
        args = dict(base_args)
        args["raw_data"] = np.array(example)
        res = generate_features(**args)
        f = res.get("features")
        if isinstance(f, np.ndarray):
            f = f.tolist()
        features_out.append(f)

        if idx == 0:
            labels = res.get("labels", [])
            output_config = res.get("output_config")

    _send_json(
        handler,
        200,
        {
            "success": True,
            "features": features_out,
            "labels": labels,
            "output_config": output_config,
        },
    )


def _send_json(handler, code, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        params = _read_parameters_json()

        if url.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            title = params["info"].get("title", "Edge Impulse DSP block")
            author = params["info"].get("author", "unknown")
            self.wfile.write(f"Edge Impulse DSP block: {title} by {author}".encode())
        elif url.path == "/parameters":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # Studio expects a "version" top-level for parameters endpoint
            params["version"] = 1
            self.wfile.write(json.dumps(params).encode())
        else:
            self.send_error(404, "Invalid path")

    def do_POST(self):
        url = urlparse(self.path)
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))

            if url.path == "/run":
                _single_request(self, body)
            elif url.path == "/batch":
                _batch_request(self, body)
            else:
                self.send_error(404, "Invalid path")
        except Exception as e:
            # Always return JSON on errors for Studio
            _send_json(self, 200, {"success": False, "error": str(e)})

    # Keep logs quiet
    def log_message(self, fmt, *args):
        return


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "4446"))
    server = ThreadingSimpleServer((host, port), Handler)
    print("Listening on host", host, "port", port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run()
