import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Making an output folder for our results
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 2: Downloading necessary NLP tools (Stopwords)
print("Downloading NLP components...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 3: Loading the Dataset
# Use a 5,000 sample subset for faster teaching/training
print("Loading IMDB Review Data...")
df = pd.read_csv('data/IMDB_Dataset.csv').sample(5000, random_state=42)

# Step 4: Text Cleaning (Important Intermediate Step!)
# Computers hate HTML tags and punctuation, so we clean them out.
def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'<br />', ' ', text) # Remove HTML tags
    text = re.sub(r'[^a-z ]', '', text) # Remove punctuation/numbers
    return text

print("Cleaning text data... this is a crucial NLP step.")
df['review'] = df['review'].apply(clean_text)

# Step 5: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# --- TASK 4: INTERMEDIATE STEPS (TF-IDF Vectorization) ---
# This is the "Magic" step where words become numbers
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- TASK 5: MODEL TRAINING (Linear Support Vector Machine) ---
# LinearSVC is much faster than standard SVC for high-dimensional text
model = LinearSVC(C=1.0, random_state=42)
print("Training the Sentiment Classifier...")
model.fit(X_train_tfidf, y_train)

# --- TASK 6: VISUALIZING THE "WORDS THAT MATTER" ---
# This is our 'Creative Teaching Tool' - showing which words drive the model
print("Identifying the most influential words...")
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_.flatten()

# Get top 10 positive and top 10 negative words
top_pos = np.argsort(coefficients)[-10:]
top_neg = np.argsort(coefficients)[:10]

plt.figure(figsize=(12, 6))
plt.barh(feature_names[top_pos], coefficients[top_pos], color='#2ca02c', label='Positive Indicators')
plt.barh(feature_names[top_neg], coefficients[top_neg], color='#d62728', label='Negative Indicators')
plt.title("Tutorial: The Top 20 Words Driving Sentiment Predictions", fontsize=14)
plt.xlabel("Coefficient Weight (Importance)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
plt.show()

# Final Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"Success! Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Confusion Matrix for Error Analysis
plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Purples')
plt.title("Final Result: Sentiment Confusion Matrix")
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
plt.show()