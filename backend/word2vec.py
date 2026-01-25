"""
Created on Fri Nov 28 11:55:58 2025

@author: Akilesh S
"""

import pandas as pd
import re
import math
import random
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def softmax_np(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    exps = np.exp(x - m)
    return exps / np.sum(exps)

def split_with_regex(s: str) -> List[str]:
    # letters or digits only, already lowercase upstream
    return re.findall(r'[a-z]+|[0-9]+', s)

def clean_title(t: str) -> str:
    # keep letters/digits/space, collapse spaces
    s = re.sub(r'[^A-Za-z0-9\s]', ' ', str(t))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# --------------------
# Word2Vec (CBOW) - NumPy optimized
# --------------------
class Word2Vec:
    def __init__(self, vocab, embedding_dim=50, learning_rate=0.05, seed: int = 42):
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        np.random.seed(seed)

        # W1: vocab_size x embedding_dim
        self.W1 = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
        # W2: embedding_dim x vocab_size
        self.W2 = np.random.uniform(-0.5, 0.5, (self.embedding_dim, self.vocab_size))

        # for tracking history
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        self.best_epoch = None
        self.best_val_loss = None

    def forward(self, context_indices: List[int]):
        if not context_indices:
            hidden = np.zeros(self.embedding_dim, dtype=float)
        else:
            context_indices = np.array(context_indices, dtype=int)
            context_embs = self.W1[context_indices]  # (C, D)
            hidden = np.mean(context_embs, axis=0)   # (D,)

        # Output layer: (D,) @ (D, V) -> (V,)
        scores = hidden @ self.W2
        output = softmax_np(scores)
        return hidden, output

    def backward(self, context_indices: List[int], target_idx: int,hidden: np.ndarray, output: np.ndarray):
        lr = self.learning_rate

        # Error = prediction - target_one_hot
        error = output.copy()
        error[target_idx] -= 1.0  # (V,)

        # Gradient W2 = outer(hidden, error)
        grad_W2 = np.outer(hidden, error)  # (D, V)

        # Gradient hidden = W2 @ error
        grad_hidden = self.W2 @ error      # (D,)

        # Update W2
        self.W2 -= lr * grad_W2

        # Update W1 only for context words
        if context_indices:
            context_indices = np.array(context_indices, dtype=int)
            self.W1[context_indices] -= (lr / len(context_indices)) * grad_hidden

    def _compute_loss_and_accuracy(self,data: List[Tuple[List[int], int]]):
        total_loss = 0.0
        correct = 0
        total = 0

        for context, target in data:
            _, output = self.forward(context)
            total_loss -= math.log(output[target] + 1e-12)
            pred_idx = int(np.argmax(output))
            if pred_idx == target:
                correct += 1
            total += 1

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(self,
              data: List[Tuple[List[int], int]],
              val_data: Optional[List[Tuple[List[int], int]]] = None,
              epochs: int = 10,
              verbose: bool = True,
              shuffle: bool = True,
              early_stopping: bool = True,
              patience: int = 3):
        best_val_loss = float("inf")
        best_epoch = -1
        no_improve = 0

        # For checkpointing best weights
        best_W1 = None
        best_W2 = None

        for epoch in range(1, epochs + 1):
            if shuffle:
                random.shuffle(data)

            total_loss = 0.0
            correct = 0
            total = 0

            for context, target in data:
                hidden, output = self.forward(context)
                self.backward(context, target, hidden, output)
                total_loss -= math.log(output[target] + 1e-12)
                pred_idx = int(np.argmax(output))
                if pred_idx == target:
                    correct += 1
                total += 1

            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            val_loss = None
            val_acc = None

            if val_data is not None and len(val_data) > 0:
                val_loss, val_acc = self._compute_loss_and_accuracy(val_data)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_improve = 0

                    best_W1 = self.W1.copy()
                    best_W2 = self.W2.copy()
                else:
                    no_improve += 1

                if early_stopping and no_improve >= patience:
                    if verbose:
                        print(
                            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
                            f"Val Acc: {val_acc:.4f}"
                        )
                        print(
                            f"Early stopping at epoch {epoch} "
                            f"(best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})"
                        )
                    # log this last epoch too
                    self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
                    break

            self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            if verbose:
                if val_loss is not None:
                    print(
                        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f}"
                    )

        if best_W1 is not None and best_W2 is not None:
            self.W1 = best_W1
            self.W2 = best_W2
            self.best_epoch = best_epoch
            self.best_val_loss = best_val_loss
            if verbose:
                print(f"Restored best model weights from epoch {best_epoch} (val loss: {best_val_loss:.4f})")

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

    def get_word_vector(self, word: str):
        idx = self.word2idx.get(word)
        if idx is None:
            return None
        return self.W1[idx]

    def get_embeddings(self):
        return {self.idx2word[i]: self.W1[i] for i in range(self.vocab_size)}
    
    def save_word_embeddings_to_csv(self, filename='word_embeddings.csv'):
        """Save all word embeddings to a CSV file"""
        rows = []
        for i, word in enumerate(self.vocab):
            vec = self.W1[i]
            row = {'word': word}
            for j, value in enumerate(vec):
                row[f'emb_{j}'] = float(value)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Saved {len(rows)} word embeddings to {filename}")
        return df

    def most_similar(self, query_word: str, top_k: int = 10):
        """Return top_k most similar words to query_word based on cosine similarity."""
        if query_word not in self.word2idx:
            print(f"'{query_word}' not in vocabulary.")
            return []

        q_vec = self.get_word_vector(query_word)
        sims = []
        for w in self.vocab:
            if w == query_word:
                continue
            w_vec = self.get_word_vector(w)
            sims.append((w, cosine_similarity(q_vec, w_vec)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

def build_dataset_from_corpus(tokenized_titles,
                              window_size: int = 5,
                              min_count: int = 1):
    # Count frequencies
    freq = Counter()
    for tokens in tokenized_titles:
        freq.update(tokens)

    # Filter rare words
    vocab = sorted([w for w, c in freq.items() if c >= min_count])
    word2idx = {w: i for i, w in enumerate(vocab)}

    data = []
    for tokens in tokenized_titles:
        # keep only words in vocab
        tokens = [w for w in tokens if w in word2idx]
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


def plot_training_curves(history, best_epoch=None):
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    # Filter out None for val metrics if no validation
    has_val = any(v is not None for v in val_loss)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    if has_val:
        plt.plot(epochs, [v if v is not None else float('nan') for v in val_loss],
                 label="Val Loss")
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CBOW Word2Vec - Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")
    if has_val:
        plt.plot(epochs, [v if v is not None else float('nan') for v in val_acc],
                 label="Val Acc")
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CBOW Word2Vec - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Path to your dataset
    INPUT_PATH = "titles_cleaned.csv"
    TITLE_COL_NAME = None  

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


    window_size = 4
    embedding_dim = 50
    learning_rate = 0.025
    epochs = 35
    min_count = 3

    # Build dataset (CBOW)
    vocab, all_pairs = build_dataset_from_corpus(
        tokenized_titles,
        window_size=window_size,
        min_count=min_count
    )

    print(f"Vocab size: {len(vocab)} | Total pairs: {len(all_pairs)}")

    # Train/validation split
    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(0.9 * len(all_pairs))
    train_data = all_pairs[:split_idx]
    val_data = all_pairs[split_idx:] if split_idx < len(all_pairs) else []

    if val_data:
        print(f"Training pairs: {len(train_data)} | Validation pairs: {len(val_data)}")
    else:
        print(f"Training pairs: {len(train_data)} | No validation set.")

    # Train Word2Vec
    model = Word2Vec(vocab, embedding_dim=embedding_dim, learning_rate=learning_rate)
    model.train(
        train_data,
        val_data=val_data,
        epochs=epochs,
        verbose=True,
        shuffle=True,
        early_stopping=True,
        patience=3
    )

    # --------- Visualize training curves ----------
    plot_training_curves(model.history, best_epoch=model.best_epoch)

    # --------- Qualitative check: nearest neighbors ----------
    test_words = ["python", "data", "love", "music"]  # change based on your vocab
    for w in test_words:
        print(f"\nMost similar to '{w}':")
        sims = model.most_similar(w, top_k=10)
        for other, score in sims:
            print(f"  {other:<15} {score:.4f}")

    # --------- Save embeddings ----------
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
