import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from collections import Counter
from pathlib import Path

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

titles_path = DATA_DIR / "titles_clean.csv"
vocab_path = DATA_DIR / "full_labeled_words.csv"
model_path = MODEL_DIR / "study_model.pkl"

# Load data
titles_df = pd.read_csv(titles_path)
titles_df['title'] = titles_df['title'].fillna('')

vocab_df = pd.read_csv(vocab_path)
vocab_dict = dict(zip(vocab_df['word'], vocab_df['label']))

# Pseudo-label titles
def pseudo_label_title(title):
    if not title:
        return 0
    tokens = word_tokenize(title.lower())
    words = [w for w in tokens if w.isalpha() and w in vocab_dict]
    if not words:
        return 0
    labels = [vocab_dict[w] for w in words]
    majority = Counter(labels).most_common(1)[0][0]
    return majority

titles_df['label'] = titles_df['title'].apply(pseudo_label_title)
labeled_df = titles_df[titles_df['label'].notna()]

print(f"Pseudo-labeled {len(labeled_df)} / {len(titles_df)} titles")
print(f"Label distribution: {labeled_df['label'].value_counts().to_dict()}")

if len(labeled_df) < 10:
    raise ValueError("Too few labeled titles. Add more data or improve labeled vocabulary.")

# Train model
vocab_list = vocab_df['word'].astype(str).tolist()
vectorizer = CountVectorizer(vocabulary=vocab_list, lowercase=True, token_pattern=r'(?u)\\b\\w+\\b')

X = vectorizer.fit_transform(labeled_df['title'])
y = labeled_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['Non-Study', 'Study']))

# Save model
with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print(f"\nModel saved at: {model_path}")
