import random
import math
import re
import pandas as pd
import pickle
from word2vec import Word2Vec, split_with_regex, clean_title

def vector_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vector_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def scalar_multiply(scalar, vec):
    return [scalar * x for x in vec]

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1

def sigmoid_derivative(y):
    return y * (1 - y)

def tanh(x):
    try:
        return math.tanh(x)
    except OverflowError:
        return -1 if x < 0 else 1

def tanh_derivative(y):
    return 1 - y * y

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def outer_product(vec1, vec2):
    rows, cols = len(vec1), len(vec2)
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = vec1[i] * vec2[j]
    return result

def preprocess_text(text):
    if pd.isna(text):
        return []
    # Convert to lowercase and remove extra spaces
    text = str(text).lower().strip()
    # Keep words, numbers, and basic punctuation
    tokens = re.findall(r"[a-zA-Z]+|[0-9]+|[.!?]", text)
    return tokens

def build_dataset_from_dataframe(df, window_size, w2v_word2idx):
    data = []
    for _, row in df.iterrows():
        words = preprocess_text(row['title'])
        # Convert labels to binary: 'study' = 1, 'non-study' = 0
        sentiment = 1.0 if row['label'] == 'study' else 0.0
        
        # Convert words to indices using Word2Vec vocabulary
        indices = [w2v_word2idx.get(word, 0) for word in words[:window_size]]  # 0 is <UNK>
        if len(indices) < window_size:
            indices += [0] * (window_size - len(indices))  # Pad with <UNK>
        data.append((indices, sentiment))
    return data

def load_word2vec_model(model_path='word2vec_model.pkl'):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['vocab'], data['word2idx'], data['embedding_dim']

def load_word_embeddings_from_csv(csv_path='word_embeddings.csv'):
    df = pd.read_csv(csv_path)
    
    # Extract word to index mapping
    word2idx = {word: idx for idx, word in enumerate(df['word'].values)}
    
    # Extract embedding vectors
    embedding_columns = [col for col in df.columns if col.startswith('emb_')]
    embedding_dim = len(embedding_columns)
    
    embeddings = []
    for _, row in df.iterrows():
        vector = [row[col] for col in embedding_columns]
        embeddings.append(vector)
    
    vocab = df['word'].tolist()
    
    print(f"Loaded {len(embeddings)} word embeddings from {csv_path}")
    print(f"Embedding dimension: {embedding_dim}")
    
    return embeddings, vocab, word2idx, embedding_dim

def create_embeddings_matrix(w2v_model, w2v_vocab_size, embedding_dim):
    embeddings = w2v_model.W1  # This is already vocab_size x embedding_dim
    return embeddings

class LSTM:
    def __init__(self, vocab_size, hidden_dim=64, input_dim=100, learning_rate=0.01, dropout_rate=0.3):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim 
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.initial_lr = learning_rate
        self.training = False

        concat_dim = input_dim + hidden_dim
        scale = math.sqrt(2.0 / (input_dim + hidden_dim))
        
        self.Wf = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(concat_dim)]
        self.bf = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.Wi = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(concat_dim)]
        self.bi = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.Wc = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(concat_dim)]
        self.bc = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.Wo = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(concat_dim)]
        self.bo = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        
        self.Wy = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.by = random.uniform(-scale, scale)

    def apply_dropout(self, vector, dropout_rate):
        if dropout_rate > 0 and self.training:
            return [x * (1 - dropout_rate) if random.random() > dropout_rate else 0.0 
                   for x in vector]
        return vector

    def forward(self, inputs, embeddings, training=False):
        self.training = training
        
        if not inputs:
            return [], 0.5
            
        h_t = [0.0] * self.hidden_dim
        c_t = [0.0] * self.hidden_dim
        states = []

        for x_idx in inputs:
            if x_idx < 0 or x_idx >= len(embeddings):
                x_t = [0.0] * self.input_dim
            else:
                x_t = embeddings[x_idx]
            
            combined = h_t + x_t

            f_t_raw = vector_add(matrix_multiply([combined], self.Wf)[0], self.bf)
            f_t = [sigmoid(x) for x in f_t_raw]

            i_t_raw = vector_add(matrix_multiply([combined], self.Wi)[0], self.bi)
            i_t = [sigmoid(x) for x in i_t_raw]

            c_tilde_raw = vector_add(matrix_multiply([combined], self.Wc)[0], self.bc)
            c_tilde = [tanh(x) for x in c_tilde_raw]

            c_t = [f * c + i * c_tilde for f, c, i, c_tilde in zip(f_t, c_t, i_t, c_tilde)]

            o_t_raw = vector_add(matrix_multiply([combined], self.Wo)[0], self.bo)
            o_t = [sigmoid(x) for x in o_t_raw]

            h_t = [o * tanh(c) for o, c in zip(o_t, c_t)]
            
            if training:
                h_t = self.apply_dropout(h_t, self.dropout_rate)

            states.append((h_t.copy(), c_t.copy(), f_t.copy(), i_t.copy(), c_tilde.copy(), o_t.copy(), combined.copy()))

        # Final output (binary classification probability)
        y_t_raw = sum(h * w for h, w in zip(h_t, self.Wy)) + self.by
        y_t = sigmoid(y_t_raw)  # Probability between 0 and 1
        return states, y_t

    def backward(self, states, inputs, target, embeddings):
        if not states[0]:  # Empty sequence
            return {}

        dh_next = [0.0] * self.hidden_dim
        dc_next = [0.0] * self.hidden_dim
        grad_embeddings = {i: [0.0] * self.input_dim for i in range(self.vocab_size)}

        states_list, y_t = states
        
        dy = (y_t - target)

        h_t_last = states_list[-1][0]
        grad_Wy = [dy * h for h in h_t_last]
        grad_by = dy

        for t in reversed(range(len(inputs))):
            h_t, c_t, f_t, i_t, c_tilde, o_t, combined = states_list[t]

            do = [dh_next[j] * tanh(c_t[j]) * sigmoid_derivative(o_t[j]) for j in range(self.hidden_dim)]
            dc = [dh_next[j] * o_t[j] * tanh_derivative(tanh(c_t[j])) + dc_next[j] for j in range(self.hidden_dim)]
            dc_tilde = [dc[j] * i_t[j] * tanh_derivative(c_tilde[j]) for j in range(self.hidden_dim)]
            di = [dc[j] * c_tilde[j] * sigmoid_derivative(i_t[j]) for j in range(self.hidden_dim)]
            df = [dc[j] * (c_t[j] if t > 0 else 0) * sigmoid_derivative(f_t[j]) for j in range(self.hidden_dim)]

            # Weight gradients
            grad_Wf = outer_product(combined, df)
            grad_Wi = outer_product(combined, di)
            grad_Wc = outer_product(combined, dc_tilde)
            grad_Wo = outer_product(combined, do)
            grad_bf = df
            grad_bi = di
            grad_bc = dc_tilde
            grad_bo = do

            # Update weights
            for i in range(len(combined)):
                for j in range(self.hidden_dim):
                    self.Wf[i][j] -= self.learning_rate * grad_Wf[i][j]
                    self.Wi[i][j] -= self.learning_rate * grad_Wi[i][j]
                    self.Wc[i][j] -= self.learning_rate * grad_Wc[i][j]
                    self.Wo[i][j] -= self.learning_rate * grad_Wo[i][j]
            
            # Update biases
            for j in range(self.hidden_dim):
                self.bf[j] -= self.learning_rate * grad_bf[j]
                self.bi[j] -= self.learning_rate * grad_bi[j]
                self.bc[j] -= self.learning_rate * grad_bc[j]
                self.bo[j] -= self.learning_rate * grad_bo[j]

            # Backpropagate to combined input
            dcombined = [0.0] * len(combined)
            for i in range(self.hidden_dim):
                for j in range(len(combined)):
                    dcombined[j] += (df[i] * self.Wf[j][i] + di[i] * self.Wi[j][i] +
                                     dc_tilde[i] * self.Wc[j][i] + do[i] * self.Wo[j][i])

            # Update embedding gradients
            if 0 <= inputs[t] < self.vocab_size:
                for j in range(self.input_dim):
                    grad_embeddings[inputs[t]][j] += dcombined[self.hidden_dim + j]

            # Prepare for next time step
            dh_next = [0.0] * self.hidden_dim
            for i in range(self.hidden_dim):
                for j in range(self.hidden_dim):
                    dh_next[i] += (df[j] * self.Wf[i][j] + di[j] * self.Wi[i][j] +
                                   dc_tilde[j] * self.Wc[i][j] + do[j] * self.Wo[i][j])

            dc_next = [dc[j] * f_t[j] for j in range(self.hidden_dim)]

        # Update output layer weights
        for i in range(self.hidden_dim):
            self.Wy[i] -= self.learning_rate * grad_Wy[i]
        self.by -= self.learning_rate * grad_by

        return grad_embeddings

    def train(self, data, epochs=50, embeddings=None, validation_data=None):
        """Train the LSTM model"""
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(data)  # Shuffle for better training
            
            # Learning rate decay
            self.learning_rate = self.initial_lr * (0.95 ** (epoch // 10))
            
            for inputs, target in data:
                states, y_t = self.forward(inputs, embeddings, training=True)
                
                # Binary cross-entropy loss
                loss = - (target * math.log(y_t + 1e-8) + (1 - target) * math.log(1 - y_t + 1e-8))
                total_loss += loss
                
                grad_embeddings = self.backward((states, y_t), inputs, target, embeddings)
                
                # Update embeddings (fine-tune during LSTM training)
                for idx, grad in grad_embeddings.items():
                    if 0 <= idx < len(embeddings):
                        for j in range(self.input_dim):
                            embeddings[idx][j] -= self.learning_rate * grad[j]
            
            # Validation
            if validation_data and epoch % 1 == 0:
                val_loss = 0.0
                correct_predictions = 0
                for inputs, target in validation_data:
                    _, y_t = self.forward(inputs, embeddings, training=False)
                    loss = - (target * math.log(y_t + 1e-8) + (1 - target) * math.log(1 - y_t + 1e-8))
                    val_loss += loss
                    
                    # Count correct predictions (threshold 0.5)
                    prediction = 1 if y_t > 0.5 else 0
                    if prediction == target:
                        correct_predictions += 1
                
                accuracy = correct_predictions / len(validation_data) if validation_data else 0
                print(f"Epoch {epoch}, Train Loss: {total_loss/len(data):.4f}, Val Loss: {val_loss/len(validation_data):.4f}, Val Acc: {accuracy:.2f}")
            elif epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(data):.4f}")

    def predict(self, inputs, embeddings):
        _, y_t = self.forward(inputs, embeddings, training=False)
        return y_t

    def predict_class(self, inputs, embeddings, threshold=0.5):
        probability = self.predict(inputs, embeddings)
        return 1 if probability > threshold else 0

    def save_model(self, filepath='lstm_model.pkl'):
        model_data = {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'initial_lr': self.initial_lr,
            'Wf': self.Wf,
            'bf': self.bf,
            'Wi': self.Wi,
            'bi': self.bi,
            'Wc': self.Wc,
            'bc': self.bc,
            'Wo': self.Wo,
            'bo': self.bo,
            'Wy': self.Wy,
            'by': self.by
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"LSTM model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath='lstm_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls(
            vocab_size=model_data['vocab_size'],
            hidden_dim=model_data['hidden_dim'],
            input_dim=model_data['input_dim'],
            learning_rate=model_data['learning_rate'],
            dropout_rate=model_data['dropout_rate']
        )
        
        # Restore weights and parameters
        model.initial_lr = model_data['initial_lr']
        model.Wf = model_data['Wf']
        model.bf = model_data['bf']
        model.Wi = model_data['Wi']
        model.bi = model_data['bi']
        model.Wc = model_data['Wc']
        model.bc = model_data['bc']
        model.Wo = model_data['Wo']
        model.bo = model_data['bo']
        model.Wy = model_data['Wy']
        model.by = model_data['by']
        
        print(f"LSTM model loaded from {filepath}")
        return model

def evaluate_model(model, val_data, embeddings, val_df):
    """Evaluate the model on validation data"""
    correct_predictions = 0
    total_predictions = len(val_data)
    
    print("\nValidation Set Predictions:")
    print("-" * 80)
    print(f"{'Title':<50} {'True':<10} {'Pred':<10} {'Prob':<10} {'Status':<10}")
    print("-" * 80)
    
    for i, (inputs, true_label) in enumerate(val_data[:20]):  # Show first 20
        probability = model.predict(inputs, embeddings)
        predicted_class = model.predict_class(inputs, embeddings)
        status = "CORRECT" if predicted_class == true_label else "WRONG"
        
        if predicted_class == true_label:
            correct_predictions += 1
            
        # Get original title
        original_title = val_df.iloc[i]['title'][:45] + "..." if len(val_df.iloc[i]['title']) > 45 else val_df.iloc[i]['title']
        
        print(f"{original_title:<50} {true_label:<10} {predicted_class:<10} {probability:.3f} {status:<10}")

    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions
    print(f"\nOverall Validation Accuracy: {accuracy:.2%}")
    
    return accuracy

def predict_single_title(model, title, w2v_word2idx, embeddings, window_size=15):
    """Predict sentiment for a single title"""
    words = preprocess_text(title)
    indices = [w2v_word2idx.get(word, 0) for word in words[:window_size]]
    if len(indices) < window_size:
        indices += [0] * (window_size - len(indices))
    
    probability = model.predict(indices, embeddings)
    predicted_class = model.predict_class(indices, embeddings)
    
    return probability, predicted_class


def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('titles_cleaned.csv')
    
    # Check class distribution
    print("Class distribution:")
    print(df['label'].value_counts())
    print(f"Total samples: {len(df)}")
    
    try:
        print("Loading pre-trained Word2Vec model...")
        w2v_model, w2v_vocab, w2v_word2idx, embedding_dim = load_word2vec_model('word2vec_model.pkl')
        print(f"Loaded Word2Vec model with vocab size: {len(w2v_vocab)}")
        print(f"Embedding dimension: {embedding_dim}")
        
        # Create embeddings matrix using pre-trained Word2Vec
        print("Creating embeddings matrix from pre-trained Word2Vec...")
        embeddings = create_embeddings_matrix(w2v_model, len(w2v_vocab), embedding_dim)
        
    except FileNotFoundError:
        print("Word2Vec model not found. Loading embeddings from CSV...")
        # Fallback: load embeddings from CSV
        embeddings, w2v_vocab, w2v_word2idx, embedding_dim = load_word_embeddings_from_csv('word_embeddings.csv')
    
    # Split data for training and validation
    train_ratio = 0.8
    train_size = int(train_ratio * len(df))
    
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Prepare LSTM datasets - using Word2Vec vocabulary directly
    print("Preparing LSTM training data...")
    train_data = build_dataset_from_dataframe(train_df, window_size=15, w2v_word2idx=w2v_word2idx)
    val_data = build_dataset_from_dataframe(val_df, window_size=15, w2v_word2idx=w2v_word2idx)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Show some examples
    print("\nSample training data:")
    for i in range(3):
        indices, label = train_data[i]
        words = [w2v_vocab[idx] if idx < len(w2v_vocab) else '<UNK>' for idx in indices[:5]]
        print(f"Sample {i+1}: Words: {words}..., Label: {'study' if label == 1 else 'non-study'}")

    # Initialize and train LSTM
    print("\nTraining LSTM model...")
    model = LSTM(
        vocab_size=len(w2v_vocab),  # Use Word2Vec vocabulary size
        hidden_dim=64, 
        input_dim=embedding_dim,  # Use Word2Vec embedding dimension
        learning_rate=0.1,
        dropout_rate=0.3
    )
    
    model.train(
        train_data, 
        epochs=20, 
        embeddings=embeddings, 
        validation_data=val_data
    )

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Evaluate model
    accuracy = evaluate_model(model, val_data, embeddings, val_df)

    # Test on some specific examples
    print("\n" + "="*60)
    print("TESTING ON SAMPLE TITLES")
    print("="*60)
    
    test_titles = [
        "Python programming tutorial for beginners",
        "Daily vlog my morning routine",
        "Machine learning study guide 2024",
        "Cooking delicious pasta recipe",
        "Math exam preparation tips"
    ]
    
    for title in test_titles:
        probability, predicted_class = predict_single_title(model, title, w2v_word2idx, embeddings)
        
        print(f"Title: {title}")
        print(f"Prediction: {'STUDY' if predicted_class == 1 else 'NON-STUDY'} (Probability: {probability:.3f})")
        print("-" * 50)

    # Save the trained LSTM model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model.save_model('lstm_model.pkl')
    
    # Also save the complete model package for easy loading
    complete_model_data = {
        'lstm_model': model,
        'w2v_vocab': w2v_vocab,
        'w2v_word2idx': w2v_word2idx,
        'embeddings': embeddings,
        'embedding_dim': embedding_dim,
        'window_size': 15
    }
    
    with open('complete_model_package.pkl', 'wb') as f:
        pickle.dump(complete_model_data, f)
    print("Complete model package saved to complete_model_package.pkl")

    # Demonstrate loading the model
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL LOADING")
    print("="*60)
    
    try:
        # Load the LSTM model
        loaded_model = LSTM.load_model('lstm_model.pkl')
        
        # Test the loaded model
        test_title = "Deep learning tutorial for advanced students"
        probability, predicted_class = predict_single_title(loaded_model, test_title, w2v_word2idx, embeddings)
        
        print(f"Loaded model prediction for: '{test_title}'")
        print(f"Result: {'STUDY' if predicted_class == 1 else 'NON-STUDY'} (Probability: {probability:.3f})")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()