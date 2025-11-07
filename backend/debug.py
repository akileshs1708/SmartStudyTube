import pickle
import os
import traceback
import sys

# Add the current directory to Python path to import Word2Vec
sys.path.append('.')

try:
    from word2vec import Word2Vec
    print("SUCCESS: Imported Word2Vec class")
except ImportError as e:
    print(f"ERROR importing Word2Vec: {e}")
    print("Trying to define a minimal Word2Vec class for loading...")
    
    # Define a minimal Word2Vec class for loading
    class Word2Vec:
        def __init__(self, vocab, embedding_dim=50, learning_rate=0.05):
            self.vocab = vocab
            self.word2idx = {w: i for i, w in enumerate(vocab)}
            self.idx2word = {i: w for i, w in enumerate(vocab)}
            self.vocab_size = len(vocab)
            self.embedding_dim = embedding_dim
            self.learning_rate = learning_rate

MODELS_DIR = "models"
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec_model.pkl")

def debug_pickle_file(filepath, name):
    print(f"\n=== Debugging {name} ===")
    print(f"File path: {filepath}")
    print(f"File exists: {os.path.exists(filepath)}")
    
    if not os.path.exists(filepath):
        print("FILE NOT FOUND!")
        return None
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        print(f"SUCCESS: Loaded {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, '__len__'):
                    if isinstance(value, (list, tuple)):
                        print(f"    Length: {len(value)}")
                    elif hasattr(value, 'shape'):
                        print(f"    Shape: {value.shape}")
                # Show sample of first few elements for lists/arrays
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float, str)):
                        print(f"    Sample: {value[:3]}")
                    elif isinstance(value[0], list):
                        print(f"    Sample shape: {len(value[0])}D vectors")
        else:
            print(f"Object type: {type(data)}")
            print(f"Available attributes: {[attr for attr in dir(data) if not attr.startswith('_')]}")
            if hasattr(data, 'vocab_size'):
                print(f"  vocab_size: {data.vocab_size}")
            if hasattr(data, 'embedding_dim'):
                print(f"  embedding_dim: {data.embedding_dim}")
            if hasattr(data, 'word2idx'):
                print(f"  word2idx: {len(data.word2idx)} entries")
            if hasattr(data, 'W1'):
                print(f"  W1 shape: {len(data.W1)} x {len(data.W1[0]) if data.W1 else 0}")
        
        return data
        
    except Exception as e:
        print(f"ERROR loading {name}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Debugging model files...")
    lstm_data = debug_pickle_file(LSTM_MODEL_PATH, "LSTM Model")
    w2v_data = debug_pickle_file(WORD2VEC_MODEL_PATH, "Word2Vec Model")