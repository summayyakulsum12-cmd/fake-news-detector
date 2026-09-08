"""
Flask app exposing:
  GET  /              -> simple web UI
  POST /predict        -> form-based prediction (used by the web UI)
  POST /api/predict    -> JSON API:  {"text": "..."}  ->  prediction JSON
  GET  /health          -> health check for deployment platforms
"""
import os

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify

from src import hybrid_predictor, ml_model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", ml_ready=ml_model.is_ready())


@app.route("/predict", methods=["POST"])
def predict_form():
    text = request.form.get("text", "")
    try:
        result = hybrid_predictor.predict(text)
        return render_template("index.html", ml_ready=ml_model.is_ready(), result=result, input_text=text)
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", ml_ready=ml_model.is_ready(), error=str(exc), input_text=text)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    try:
        result = hybrid_predictor.predict(text)
        return jsonify({"success": True, "result": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ml_model_ready": ml_model.is_ready()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
