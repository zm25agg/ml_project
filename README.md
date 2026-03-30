# Natural Language Processing: Decoding Sentiment with TF-IDF and LinearSVC

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
This repository contains a professional-standard tutorial on **Natural Language Processing (NLP)**. The project demonstrates how to transform unstructured human language into a high-dimensional mathematical space to perform sentiment analysis on movie reviews.

By combining **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency) and **Linear Support Vector Classification (LinearSVC)**, we achieve an 83.5% accuracy in distinguishing between positive and negative sentiments in a sample of 5,000 IMDB reviews.

## 🚀 Key Technical Features
- **NLP Text Pipeline:** Implementing a robust cleaning process using Regular Expressions (`re`) and the NLTK library to handle HTML tags, punctuation, and stop-words.
- **Feature Engineering (TF-IDF):** Converting text into a 2,500-dimensional sparse matrix while mathematically penalizing frequent but uninformative words.
- **High-Performance Classification:** Utilizing LinearSVC for efficient and accurate decision-making in high-dimensional text space.
- **Model Transparency:** Visualizing the specific "Informative Words" that drive the model's positive and negative predictions.

## 📁 Project Structure
```text
NLP-Sentiment-Tutorial/
├── data/                   <-- kaggle dataset
├── output/                 <-- plots
├── main.py                 <-- main code
├── notebook/               <-- notebook
├── report/                 <-- report
├── README.md              
├── requirements.txt        <-- Dependencies
└── LICENSE                 <-- MIT License
```

## 📊 Visual Results
The tutorial generates two primary visualizations (saved in the `/output` folder):
1. **The Word Influence Chart:** A professional bar chart showing the top 10 positive and 10 negative words that the model "learned" to associate with sentiment.
2. **Sentiment Confusion Matrix:** A purple-scale visualization demonstrating the model's precision and recall across 1,000 test reviews.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/zm25agg/NLP-Sentiment-Tutorial.git
   cd NLP-Sentiment-Tutorial
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data:**
   Download the [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB_Dataset.csv` into the `/data` folder.

## ♿ Accessibility & Inclusive Design
In compliance with the assignment rubric, this project emphasizes inclusivity:
- **Visual Contrast:** We use a high-contrast Red/Green color palette for feature importance and a Purple sequential scale for the confusion matrix.
- **Documented Code:** Comments are written in clear, simple English to ensure the tutorial functions as an effective teaching tool for all levels of developers.

## ⚖️ Ethical AI: Bias in Language
NLP models are susceptible to "Linguistic Bias." This tutorial addresses the ethical responsibility to audit training data for cultural and demographic biases, ensuring that the model does not misinterpret specific dialects or slang as "negative sentiment."

## 📚 References
1. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* (Documentation for TF-IDF and LinearSVC).
2. **Bird, S., Klein, E., & Loper, E. (2009).** *Natural Language Processing with Python.* (The foundation for NLTK text processing).
3. **Kaggle Dataset:** *IMDB Dataset of 50K Movie Reviews* by Lakshmipathi N.

## 📄 License
This project is licensed under the **MIT License**.
