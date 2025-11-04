import importlib
import os
from typing import Any, Callable, Optional


def _try_import(module_name: str):
	try:
		return importlib.import_module(module_name)
	except Exception:
		return None


def _find_callable(module, candidates: list[str]) -> Optional[Callable[..., Any]]:
	if module is None:
		return None
	for name in candidates:
		func = getattr(module, name, None)
		if callable(func):
			return func
	return None


def predict(input_payload: dict) -> Any:
	"""Attempts to call a function from bsit_runner or bsit_recommendation.

	Order of preference per module: predict -> run -> main
	Passes the full JSON payload through as a single argument.
	If no function is found, returns a 501-like dict.
	"""

	# Make MODEL_PATH available to user code if they rely on it
	# Default to ./models/rf_ict.pkl (place your file there or set env var)
	model_path = os.environ.get("MODEL_PATH")
	if not model_path:
		default_path = os.path.join(os.path.dirname(__file__), "models", "rf_ict.pkl")
		os.environ["MODEL_PATH"] = default_path

	bsit_runner = _try_import("bsit_runner")
	bsit_reco = _try_import("bsit_recommendation")

	call_order = ["predict", "run", "main"]

	func = _find_callable(bsit_runner, call_order) or _find_callable(bsit_reco, call_order)
	if func is None:
		return {
			"error": "No callable entrypoint found.",
			"details": "Expected one of predict/run/main in bsit_runner or bsit_recommendation.",
			"hint": "Ensure your files are in the same folder as app.py and exported functions accept a single dict arg.",
		}

	return func(input_payload)


