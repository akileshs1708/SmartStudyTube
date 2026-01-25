from flask import Flask, request, jsonify
import pickle
import os
import re
import traceback

from lstm import LSTM
from word2vec import Word2Vec

app = Flask(__name__)

MODELS_DIR = "models"
COMPLETE_MODEL_PATH = os.path.join(MODELS_DIR, "complete_model_package.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec_model.pkl")

lstm_model = None
w2v_word2idx = None
embeddings = None
embedding_dim = None
window_size = 12
models_loaded = False

def preprocess_text(text):
    if text is None:
        return []
    text = str(text).lower().strip()
    return re.findall(r"[a-zA-Z]+|[0-9]+|[.!?]", text)

def load_models():
    global lstm_model, w2v_word2idx, embeddings, embedding_dim, window_size, models_loaded
    if os.path.exists(COMPLETE_MODEL_PATH):
        try:
            print("Loading complete_model_package.pkl ...")
            with open(COMPLETE_MODEL_PATH, "rb") as f:
                data = pickle.load(f)

            lstm_model = data["lstm_model"]
            w2v_word2idx = data["w2v_word2idx"]
            embeddings = data["embeddings"]
            embedding_dim = data["embedding_dim"]
            window_size = data.get("window_size", 12)

            models_loaded = True
            print("Loaded complete model package")
            return True

        except Exception as e:
            print("Failed loading complete package:", e)
            traceback.print_exc()

    try:
        print("Falling back to individual model files...")

        # Load Word2Vec
        with open(WORD2VEC_MODEL_PATH, "rb") as f:
            w2v_data = pickle.load(f)

        if isinstance(w2v_data, dict):
            w2v_word2idx = w2v_data["word2idx"]
            embeddings = w2v_data["model"].W1
            embedding_dim = w2v_data["embedding_dim"]
        else:
            w2v_word2idx = w2v_data.word2idx
            embeddings = w2v_data.W1
            embedding_dim = w2v_data.embedding_dim

        # Load LSTM
        lstm_model = LSTM.load_model(LSTM_MODEL_PATH)

        models_loaded = True
        print("Loaded individual LSTM + Word2Vec models")
        return True

    except Exception as e:
        print("Failed loading individual models:", e)
        traceback.print_exc()
        return False

def predict_title(title):
    words = preprocess_text(title)
    indices = [w2v_word2idx.get(w, 0) for w in words[:window_size]]

    if len(indices) < window_size:
        indices += [0] * (window_size - len(indices))

    probability = lstm_model.predict(indices, embeddings)
    prediction = 1 if probability > 0.5 else 0

    return probability, prediction, words


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "SmartStudyTube Backend",
        "models_loaded": models_loaded,
        "endpoints": {
            "/predict": "POST {title}",
            "/batch_predict": "POST {titles: []}",
            "/health": "GET"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "models_loaded": models_loaded,
        "vocab_size": len(w2v_word2idx) if w2v_word2idx else 0,
        "embedding_dim": embedding_dim,
        "window_size": window_size
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json(force=True)
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"error": "Title is required"}), 400

    prob, pred, words = predict_title(title)

    return jsonify({
        "title": title,
        "prediction": int(pred),
        "probability": round(float(prob), 4),
        "label": "Study Content" if pred else "Non-Study Content",
        "confidence": "High" if prob > 0.7 or prob < 0.3 else "Medium",
        "tokens": words[:10]
    })

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json(force=True)
    titles = data.get("titles", [])

    results = []
    for t in titles:
        if not isinstance(t, str) or not t.strip():
            continue
        prob, pred, _ = predict_title(t)
        results.append({
            "title": t,
            "prediction": int(pred),
            "probability": round(float(prob), 4),
            "label": "Study Content" if pred else "Non-Study Content"
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    print("\nStarting SmartStudyTube Backend")
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
