from __future__ import annotations

import json
import ssl
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "fraud_model.joblib"
SUMMARY_FILE = ARTIFACT_DIR / "model_summary.json"
CERT_FILE = ROOT / ".certs" / "cert.pem"
KEY_FILE = ROOT / ".certs" / "key.pem"
PORT = 8443


class FraudRequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        if not MODEL_FILE.exists():
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Model has not been trained yet")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        try:
            request_data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return

        bundle = joblib.load(MODEL_FILE)
        model = bundle["model"]
        features = bundle["features"]
        threshold = float(bundle.get("threshold", 0.5))

        try:
            values = [float(request_data[name]) for name in features]
        except (KeyError, TypeError, ValueError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing or invalid feature values")
            return

        probability = float(model.predict_proba([values])[0][1])
        prediction = "Fraud" if probability >= threshold else "Genuine"

        response = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "risk_score": round(probability * 100, 2),
        }

        encoded = json.dumps(response).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/api/model-summary":
            if not SUMMARY_FILE.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Model summary not found")
                return
            content = SUMMARY_FILE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        return super().do_GET()


def main() -> None:
    handler = partial(FraudRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"HTTPS server running at https://127.0.0.1:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
