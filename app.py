import os
from typing import List

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from runner_adapter import predict as adapter_predict


def parse_allowed_origins(env_value: str | None) -> List[str]:
	if not env_value:
		return []
	parts = [part.strip() for part in env_value.split(",")]
	return [p for p in parts if p]


app = Flask(__name__)

# Configure CORS: set ALLOWED_ORIGINS env var on Render, e.g.
# "https://yourname.epizy.com,http://yourname.epizy.com,https://your-custom-domain.com"
allowed_origins = parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))
cors_resources = {r"/*": {"origins": allowed_origins or ["*"]}}
CORS(app, resources=cors_resources)


@app.get("/health")
def health() -> tuple[dict, int]:
	return {"status": "ok"}, 200


@app.get("/api/hello")
def hello() -> dict:
	return {"message": "Hello from Render!"}


@app.post("/api/recommend")
def recommend() -> tuple[dict, int]:
	try:
		payload = request.get_json(silent=True) or {}
		result = adapter_predict(payload)
		status = 200 if not isinstance(result, dict) or "error" not in result else 501
		return (result if isinstance(result, dict) else {"result": result}, status)
	except Exception as exc:
		return ({"error": "Unhandled exception", "details": str(exc)}, 500)


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "5000"))
	app.run(host="0.0.0.0", port=port)


