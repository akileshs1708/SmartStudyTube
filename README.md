# SmartStudyTube

SmartStudyTube is a **browser extension integrated with a Machine Learning backend** that helps students stay focused while using YouTube.

It automatically analyzes video titles and **filters out distracting or non-study videos**, allowing only **educational and study-related content** to appear.

The system uses **Natural Language Processing (NLP)** and **Machine Learning models (Word2Vec + LSTM)** to classify YouTube videos as **Study** or **Non-Study**.

---

# Features

- Automatic **YouTube Video Classification**
- Filters out **non-study videos**
- Uses **Word2Vec embeddings** for text representation
- **LSTM model** for classification
- **Browser extension integration**
- Debug logs for monitoring predictions

---

# Project Structure

```
SmartStudyTube
│
├── backend
│   ├── app.py
│   ├── train_model.py
│   ├── lstm.py
│   ├── debug.py
│   ├── requirements.txt
│   │
│   ├── data
│   │   ├── full_labeled_words.csv
│   │   ├── manual_labels.csv
│   │   └── titles_clean.csv
│   │
│   └── models
│       ├── complete_model_package.pkl
│       ├── lstm_model.pkl
│       ├── model.pkl
│       ├── study_model.pkl
│       └── word2vec_model.pkl
│
└── extension
    ├── manifest.json
    ├── content.js
    ├── background.js
    ├── logs.html
    └── icon.png
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/SmartStudyTube.git
cd SmartStudyTube
```

---

# Backend Setup

### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Run the Backend Server

```bash
python app.py
```

The backend will start and handle **video classification requests from the extension**.

---

# Train the Model (Optional)

If you want to retrain the machine learning model:

```bash
python train_model.py
```

This will train the **Word2Vec + LSTM model** using the dataset located in:

```
backend/data/
```

---

# Browser Extension Setup

1. Open **Google Chrome**
2. Go to

```
chrome://extensions/
```

3. Enable **Developer Mode**
4. Click **Load Unpacked**
5. Select the folder

```
extension/
```

The SmartStudyTube extension will now be installed in your browser.

---

# How It Works

1. The **browser extension reads YouTube video titles**.
2. Titles are sent to the **backend API**.
3. The backend:
   - Cleans the text
   - Converts it to **Word2Vec embeddings**
   - Sends the vectors to the **LSTM model**
4. The model predicts whether the video is:

```
Study Content
or
Non-Study Content
```

5. Non-study videos are **hidden or filtered** from the page.

---

# Dataset

The dataset used for training includes:

- `titles_clean.csv` → cleaned video titles  
- `manual_labels.csv` → manually labeled study vs non-study data  
- `full_labeled_words.csv` → extended labeled dataset  

These datasets help train the **classification model**.

---

# Technologies Used

### Machine Learning
- Python
- Word2Vec
- LSTM
- Scikit-learn
- Pandas
- NumPy

---

# Use Cases

- Students who want to **avoid distractions on YouTube**
- Productivity tools for **focused learning**
- Educational content filtering
- Study productivity extensions

---

# Future Improvements

- Personalized study recommendations
- Firefox extension support
- Transformer-based models (BERT)
- Analytics dashboard
- Real-time learning insights

---

#  Goal

Developed as part of an **AI/ML project focused on improving student productivity through intelligent content filtering**.
