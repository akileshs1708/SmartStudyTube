from flask import Flask, request, jsonify
import pickle
import traceback
import os
import re
import numpy as np

app = Flask(__name__)

MODELS_DIR = "models"
COMPLETE_MODEL_PATH = os.path.join(MODELS_DIR, "complete_model_package.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec_model.pkl")

# Global variables for model components
lstm_model = None
w2v_word2idx = None
embeddings = None
embedding_dim = None
window_size = 15

print("Loading models...")

def preprocess_text(text):
    if text is None:
        return []
    text = str(text).lower().strip()
    tokens = re.findall(r"[a-zA-Z]+|[0-9]+|[.!?]", text)
    return tokens

def load_complete_model_package():
    global lstm_model, w2v_word2idx, embeddings, embedding_dim, window_size
    
    if not os.path.exists(COMPLETE_MODEL_PATH):
        print(f"Complete model package not found at {COMPLETE_MODEL_PATH}")
        return False
    
    try:
        print("Loading complete model package...")
        with open(COMPLETE_MODEL_PATH, "rb") as f:
            complete_data = pickle.load(f)
        
        print(f"Complete package type: {type(complete_data)}")
        
        if isinstance(complete_data, dict):
            print(f"Complete package keys: {list(complete_data.keys())}")
            
            # Extract all components from complete package
            lstm_model = complete_data.get('lstm_model')
            w2v_word2idx = complete_data.get('w2v_word2idx')
            embeddings = complete_data.get('embeddings')
            embedding_dim = complete_data.get('embedding_dim')
            window_size = complete_data.get('window_size', 15)
            
            print(f"LSTM model: {lstm_model is not None}")
            print(f"Vocabulary: {len(w2v_word2idx) if w2v_word2idx else 'None'} words")
            print(f"Embeddings: {len(embeddings) if embeddings else 'None'} vectors")
            print(f"Embedding dim: {embedding_dim}")
            print(f"Window size: {window_size}")
            
            if lstm_model and w2v_word2idx and embeddings is not None:
                print("SUCCESS: Complete model package loaded!")
                return True
            else:
                print("WARNING: Some components missing from complete package")
                return False
        else:
            print("Complete package is not a dictionary")
            return False
            
    except Exception as e:
        print(f"ERROR loading complete model package: {e}")
        traceback.print_exc()
        return False

def load_individual_models():
    """Load individual model files if complete package fails"""
    global lstm_model, w2v_word2idx, embeddings, embedding_dim
    
    try:
        # Load Word2Vec model
        print("Loading Word2Vec model...")
        if not os.path.exists(WORD2VEC_MODEL_PATH):
            print(f"Word2Vec model not found at {WORD2VEC_MODEL_PATH}")
            return False
        
        # Handle Word2Vec class dependency
        w2v_data = None
        try:
            with open(WORD2VEC_MODEL_PATH, "rb") as f:
                w2v_data = pickle.load(f)
        except (AttributeError, ModuleNotFoundError) as e:
            print(f"Word2Vec loading failed: {e}")
            # Define minimal Word2Vec class
            class MinimalWord2Vec:
                def __init__(self, vocab, embedding_dim=50, learning_rate=0.05):
                    self.vocab = vocab
                    self.word2idx = {w: i for i, w in enumerate(vocab)}
                    self.idx2word = {i: w for i, w in enumerate(vocab)}
                    self.vocab_size = len(vocab)
                    self.embedding_dim = embedding_dim
                    self.learning_rate = learning_rate
            
            import __main__
            setattr(__main__, 'Word2Vec', MinimalWord2Vec)
            
            with open(WORD2VEC_MODEL_PATH, "rb") as f:
                w2v_data = pickle.load(f)
        
        # Extract Word2Vec components
        if isinstance(w2v_data, dict):
            w2v_word2idx = w2v_data.get('word2idx')
            if 'model' in w2v_data and hasattr(w2v_data['model'], 'W1'):
                embeddings = w2v_data['model'].W1
                embedding_dim = len(embeddings[0]) if embeddings else None
            elif 'embeddings' in w2v_data:
                embeddings = w2v_data['embeddings']
                embedding_dim = len(embeddings[0]) if embeddings else None
            embedding_dim = w2v_data.get('embedding_dim', embedding_dim)
        else:
            if hasattr(w2v_data, 'word2idx'):
                w2v_word2idx = w2v_data.word2idx
            if hasattr(w2v_data, 'W1'):
                embeddings = w2v_data.W1
                embedding_dim = len(embeddings[0]) if embeddings else None
        
        print(f"Word2Vec vocab: {len(w2v_word2idx) if w2v_word2idx else 'None'}")
        
        # Load LSTM model
        print("Loading LSTM model...")
        if not os.path.exists(LSTM_MODEL_PATH):
            print(f"LSTM model not found at {LSTM_MODEL_PATH}")
            return False
        
        with open(LSTM_MODEL_PATH, "rb") as f:
            lstm_data = pickle.load(f)
        
        # Create LSTM wrapper
        class SimpleLSTM:
            def __init__(self, model_data):
                if isinstance(model_data, dict):
                    self.vocab_size = model_data.get('vocab_size', 10000)
                    self.hidden_dim = model_data.get('hidden_dim', 64)
                    self.input_dim = model_data.get('input_dim', 100)
                    self.Wf = model_data.get('Wf', [])
                    self.bf = model_data.get('bf', [])
                    self.Wi = model_data.get('Wi', [])
                    self.bi = model_data.get('bi', [])
                    self.Wc = model_data.get('Wc', [])
                    self.bc = model_data.get('bc', [])
                    self.Wo = model_data.get('Wo', [])
                    self.bo = model_data.get('bo', [])
                    self.Wy = model_data.get('Wy', [])
                    self.by = model_data.get('by', 0.0)
                else:
                    self.vocab_size = getattr(model_data, 'vocab_size', 10000)
                    self.hidden_dim = getattr(model_data, 'hidden_dim', 64)
                    self.input_dim = getattr(model_data, 'input_dim', 100)
                    self.Wf = getattr(model_data, 'Wf', [])
                    self.bf = getattr(model_data, 'bf', [])
                    self.Wi = getattr(model_data, 'Wi', [])
                    self.bi = getattr(model_data, 'bi', [])
                    self.Wc = getattr(model_data, 'Wc', [])
                    self.bc = getattr(model_data, 'bc', [])
                    self.Wo = getattr(model_data, 'Wo', [])
                    self.bo = getattr(model_data, 'bo', [])
                    self.Wy = getattr(model_data, 'Wy', [])
                    self.by = getattr(model_data, 'by', 0.0)
            
            def predict(self, inputs, embeddings):
                try:
                    h_t = [0.0] * self.hidden_dim
                    c_t = [0.0] * self.hidden_dim
                    
                    for x_idx in inputs:
                        if x_idx < 0 or x_idx >= len(embeddings):
                            x_t = [0.0] * self.input_dim
                        else:
                            x_t = embeddings[x_idx]
                        
                        combined = h_t + x_t
                        
                        f_t = [self.sigmoid(sum(combined[k] * self.Wf[k][j] for k in range(len(combined))) + self.bf[j]) 
                              for j in range(self.hidden_dim)]
                        i_t = [self.sigmoid(sum(combined[k] * self.Wi[k][j] for k in range(len(combined))) + self.bi[j]) 
                              for j in range(self.hidden_dim)]
                        c_tilde = [self.tanh(sum(combined[k] * self.Wc[k][j] for k in range(len(combined))) + self.bc[j]) 
                                  for j in range(self.hidden_dim)]
                        o_t = [self.sigmoid(sum(combined[k] * self.Wo[k][j] for k in range(len(combined))) + self.bo[j]) 
                              for j in range(self.hidden_dim)]
                        
                        c_t = [f_t[j] * c_t[j] + i_t[j] * c_tilde[j] for j in range(self.hidden_dim)]
                        h_t = [o_t[j] * self.tanh(c_t[j]) for j in range(self.hidden_dim)]
                    
                    y_t_raw = sum(h_t[j] * self.Wy[j] for j in range(self.hidden_dim)) + self.by
                    return self.sigmoid(y_t_raw)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    return 0.5
            
            def sigmoid(self, x):
                try:
                    return 1.0 / (1.0 + np.exp(-x))
                except:
                    return 0.0 if x < 0 else 1.0
            
            def tanh(self, x):
                try:
                    return np.tanh(x)
                except:
                    return -1.0 if x < 0 else 1.0
        
        lstm_model = SimpleLSTM(lstm_data)
        print("SUCCESS: Individual models loaded")
        return True
        
    except Exception as e:
        print(f"ERROR loading individual models: {e}")
        traceback.print_exc()
        return False

def load_models():
    print("Attempting to load models...")
    
    # Try complete model package first
    if load_complete_model_package():
        return True
    
    print("Falling back to individual model files...")
    
    if load_individual_models():
        return True
    
    print("All loading methods failed")
    return False

def predict_single_title(title):
    """Predict sentiment for a single title"""
    try:
        words = preprocess_text(title)
        indices = [w2v_word2idx.get(word, 0) for word in words[:window_size]]
        if len(indices) < window_size:
            indices += [0] * (window_size - len(indices))
        
        probability = lstm_model.predict(indices, embeddings)
        predicted_class = 1 if probability > 0.5 else 0
        
        return probability, predicted_class, words, indices
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.5, 0, [], []

# Load models on startup
print("\nInitializing model loading...")
models_loaded = load_models()

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET"])
def home():
    model_source = "complete_model_package.pkl" if os.path.exists(COMPLETE_MODEL_PATH) and models_loaded else "individual files"
    
    return jsonify({
        "message": "SmartStudyTube LSTM Backend is running",
        "endpoints": {
            "/predict": "POST - send {'title': '<YouTube title>'} to get prediction",
            "/health": "GET - check model status",
            "/test": "GET - test prediction with sample title"
        },
        "model_loaded": models_loaded,
        "model_source": model_source
    })

@app.route("/health", methods=["GET"])
def health():
    files_status = {
        "complete_model_package": os.path.exists(COMPLETE_MODEL_PATH),
        "lstm_model": os.path.exists(LSTM_MODEL_PATH),
        "word2vec_model": os.path.exists(WORD2VEC_MODEL_PATH)
    }
    
    status = {
        "model_loaded": models_loaded,
        "components": {
            "lstm_model": lstm_model is not None,
            "vocabulary": w2v_word2idx is not None,
            "embeddings": embeddings is not None
        },
        "vocabulary_size": len(w2v_word2idx) if w2v_word2idx else 0,
        "embedding_dim": embedding_dim,
        "window_size": window_size,
        "files_status": files_status
    }
    return jsonify(status)

@app.route("/test", methods=["GET"])
def test_prediction():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        test_title = "Python programming tutorial for beginners"
        probability, predicted_class, processed_words, _ = predict_single_title(test_title)
        
        return jsonify({
            "test_title": test_title,
            "prediction": int(predicted_class),
            "probability": float(probability),
            "label": "Study Content" if predicted_class == 1 else "Non-Study Content",
            "processed_words": processed_words,
            "status": "Test successful"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not models_loaded:
            return jsonify({"error": "Models not loaded"}), 503

        data = request.get_json(force=True)
        title = data.get("title", "").strip()

        if not title:
            return jsonify({"error": "Title not provided"}), 400

        print(f"Predicting: {title}")
        probability, predicted_class, processed_words, _ = predict_single_title(title)
        
        label = "Study Content" if predicted_class == 1 else "Non-Study Content"
        confidence = "High" if probability > 0.7 or probability < 0.3 else "Medium"

        response = {
            "title": title,
            "prediction": int(predicted_class),
            "probability": float(probability),
            "label": label,
            "confidence": confidence,
            "processed_words": processed_words[:10],
            "word_count": len(processed_words)
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        if not models_loaded:
            return jsonify({"error": "Models not loaded"}), 503

        data = request.get_json(force=True)
        titles = data.get("titles", [])
        
        if not titles or not isinstance(titles, list):
            return jsonify({"error": "Please provide a list of titles in 'titles' field"}), 400

        print(f"Processing {len(titles)} titles in batch...")
        results = []
        
        for title in titles:
            if not isinstance(title, str) or not title.strip():
                continue
                
            probability, predicted_class, processed_words, _ = predict_single_title(title.strip())
            label = "Study Content" if predicted_class == 1 else "Non-Study Content"
            
            results.append({
                "title": title.strip(),
                "prediction": int(predicted_class),
                "probability": float(probability),
                "label": label,
                "word_count": len(processed_words)
            })

        # Calculate statistics
        study_count = sum(1 for r in results if r['prediction'] == 1)
        batch_stats = {
            "total_titles": len(results),
            "study_content": study_count,
            "non_study_content": len(results) - study_count,
            "study_percentage": round((study_count / len(results)) * 100, 2) if results else 0
        }

        return jsonify({
            "results": results,
            "batch_statistics": batch_stats
        })

    except Exception as e:
        print(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting SmartStudyTube LSTM Backend")
    print("="*50)
    
    print(f"Models directory: {os.path.abspath(MODELS_DIR)}")
    print(f"Complete package: {COMPLETE_MODEL_PATH} - {'Found' if os.path.exists(COMPLETE_MODEL_PATH) else 'Missing'}")
    print(f"LSTM model: {LSTM_MODEL_PATH} - {'Found' if os.path.exists(LSTM_MODEL_PATH) else 'Missing'}")
    print(f"Word2Vec model: {WORD2VEC_MODEL_PATH} - {'Found' if os.path.exists(WORD2VEC_MODEL_PATH) else 'Missing'}")
    
    if models_loaded:
        print("\nSUCCESS: Models loaded successfully!")
        print("Server is ready to accept requests")
        
        # Quick test
        print("\nQuick test...")
        test_title = "How To Create A Login System In PHP For Beginners  Procedural MySQLi  2018 PHP Tutorial  mmtuts"
        prob, cls, words, _ = predict_single_title(test_title)
        print(f"Test: '{test_title}' -> {cls} (probability: {prob:.3f})")
    else:
        print("\nERROR: All model loading methods failed")
        print("Please ensure at least one model file exists in the models directory")
    
    print("\nStarting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)