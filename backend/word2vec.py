import pandas as pd
import re
import math
import random
from typing import List

# --------------------
# Helpers
# --------------------
def vector_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vector_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def scalar_multiply(scalar, vec):
    return [scalar * x for x in vec]

def softmax(x):
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    s = sum(exps)
    return [e / s for e in exps]

def outer_product(vec1, vec2):
    rows, cols = len(vec1), len(vec2)
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = vec1[i] * vec2[j]
    return result

def split_with_regex(s: str) -> List[str]:
    # letters or digits only, already lowercase upstream
    return re.findall(r'[a-z]+|[0-9]+', s)

def clean_title(t: str) -> str:
    # keep letters/digits/space, collapse spaces
    s = re.sub(r'[^A-Za-z0-9\s]', ' ', str(t))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --------------------
# Word2Vec (CBOW)
# --------------------
class Word2Vec:
    def __init__(self, vocab, embedding_dim=50, learning_rate=0.05):
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        random.seed(42)

        # W1: vocab_size x embedding_dim
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(embedding_dim)]
                   for _ in range(self.vocab_size)]
        # W2: embedding_dim x vocab_size
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(self.vocab_size)]
                   for _ in range(embedding_dim)]

    def forward(self, context_indices):
        hidden = [0.0] * self.embedding_dim
        for idx in context_indices:
            if 0 <= idx < self.vocab_size:
                hidden = vector_add(hidden, self.W1[idx])  # embedding lookup
        if context_indices:
            hidden = scalar_multiply(1.0 / len(context_indices), hidden)

        # Output layer (hidden @ W2)
        output_raw = [0.0] * self.vocab_size
        for j in range(self.vocab_size):
            s = 0.0
            for k in range(self.embedding_dim):
                s += self.W2[k][j] * hidden[k]
            output_raw[j] = s

        output = softmax(output_raw)
        return hidden, output

    def backward(self, context_indices, target_idx, hidden, output):
        # Error = prediction - target
        target = [0.0] * self.vocab_size
        target[target_idx] = 1.0
        error = vector_sub(output, target)

        # Gradient W2
        grad_W2 = outer_product(hidden, error)

        # Gradient hidden
        grad_hidden = [0.0] * self.embedding_dim
        for i in range(self.embedding_dim):
            s = 0.0
            for j in range(self.vocab_size):
                s += self.W2[i][j] * error[j]
            grad_hidden[i] = s

        # Update W2
        lr = self.learning_rate
        for i in range(self.embedding_dim):
            row = self.W2[i]
            grow = grad_W2[i]
            for j in range(self.vocab_size):
                row[j] -= lr * grow[j]

        # Update W1 (only for context words)
        if context_indices:
            for idx in context_indices:
                w = self.W1[idx]
                for j in range(self.embedding_dim):
                    w[j] -= lr * grad_hidden[j] / len(context_indices)

    def train(self, data, epochs=5, verbose=True):
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for context, target in data:
                hidden, output = self.forward(context)
                self.backward(context, target, hidden, output)
                total_loss -= math.log(output[target] + 1e-12)
            if verbose:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def get_word_vector(self, word: str):
        idx = self.word2idx.get(word)
        if idx is None:
            return None
        return self.W1[idx]

    def get_embeddings(self):
        return {self.idx2word[i]: self.W1[i] for i in range(self.vocab_size)}
    
    def save_word_embeddings_to_csv(self, filename='word_embeddings.csv'):
        """Save all word embeddings to a CSV file"""
        embeddings_data = []
        
        for i, word in enumerate(self.vocab):
            embedding_vector = self.W1[i]
            row = {'word': word}
            # Add each dimension of the embedding vector
            for j, value in enumerate(embedding_vector):
                row[f'emb_{j}'] = value
            embeddings_data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(embeddings_data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(embeddings_data)} word embeddings to {filename}")
        
        return df

# --------------------
# Build training data from a whole corpus (list of token lists)
# --------------------
def build_dataset_from_corpus(tokenized_titles, window_size=5):
    # Flatten all words for vocab
    words = [w for tokens in tokenized_titles for w in tokens]
    vocab = sorted(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}

    data = []
    for tokens in tokenized_titles:
        n = len(tokens)
        for i in range(n):
            target_word = tokens[i]
            context = []
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if j != i:
                    context.append(word2idx[tokens[j]])
            if context:
                data.append((context, word2idx[target_word]))
    return vocab, data

# --------------------
# Main: Load CSV, clean, tokenize, train Word2Vec
# --------------------
if __name__ == "__main__":
    # Path to your dataset
    INPUT_PATH = "titles_cleaned100.csv"
    TITLE_COL_NAME = None  # set to the exact title column name if you know it; otherwise auto-detect

    # Load
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")

    # Pick a title column (first column fallback)
    if TITLE_COL_NAME is None:
        title_col = df.columns[0]
    else:
        title_col = TITLE_COL_NAME

    # Clean titles, drop null/empty
    df["title"] = df[title_col].apply(clean_title)
    df["title"].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    # Tokenize
    tokenized_titles = [split_with_regex(t.lower()) for t in df["title"]]

    # Remove any empty token rows after tokenization
    keep_mask = [len(toks) > 0 for toks in tokenized_titles]
    df = df[keep_mask].reset_index(drop=True)
    tokenized_titles = [t for t in tokenized_titles if len(t) > 0]

    print(f"Total titles used for training: {len(tokenized_titles)}")

    # Build dataset (CBOW) and train Word2Vec
    window_size = 15
    embedding_dim = 15  # Increased for better word representations
    learning_rate = 0.2
    epochs = 50

    vocab, train_data = build_dataset_from_corpus(tokenized_titles, window_size=window_size)
    print(f"Vocab size: {len(vocab)} | Training pairs: {len(train_data)}")

    model = Word2Vec(vocab, embedding_dim=embedding_dim, learning_rate=learning_rate)
    model.train(train_data, epochs=epochs, verbose=True)
    
    print("\nSaving word embeddings to CSV...")
    embeddings_df = model.save_word_embeddings_to_csv('word_embeddings.csv')

    # Save the Word2Vec model for LSTM to use
    import pickle
    with open('word2vec_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'vocab': vocab,
            'word2idx': model.word2idx,
            'embedding_dim': embedding_dim
        }, f)
    print(f"Saved Word2Vec model to word2vec_model.pkl")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Vocabulary size: {len(vocab)}")