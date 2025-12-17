# Sentiment Analysis on IMDB Movie Reviews

This project implements **sentiment analysis classifiers** using **Logistic Regression** and **Support Vector Machines (LinearSVC)** on the IMDB movie reviews dataset provided by **NLTK**.  
The goal is to classify movie reviews as **Positive** or **Negative** using classical machine learning techniques and TF-IDF feature extraction.

---

## Overview

- Dataset: IMDB Movie Reviews (NLTK)
- Task: Binary sentiment classification
- Models:
  - Logistic Regression
  - Support Vector Machine (LinearSVC)
- Feature Representation: TF-IDF (5000 features)
- Evaluation:
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix Visualization
- Extra:
  - Interactive sentiment prediction on custom input text

---

## Features

### Text Preprocessing
Each review undergoes the following preprocessing steps:

1. Removal of:
   - Twitter-style handles (`@username`)
   - URLs
2. Tokenization using NLTK
3. Conversion to lowercase
4. Stop-word removal (English)
5. Removal of non-alphabetic tokens
6. Stemming using **Porter Stemmer**

---

### Feature Extraction (TF-IDF)

- Uses `TfidfVectorizer`
- Maximum features: **5000**
- Output:
  - Sparse, high-dimensional feature vectors
  - Suitable for linear models like Logistic Regression and SVM

---

### Classification Models

#### Logistic Regression
- Efficient linear classifier for text data
- Outputs:
  - Class predictions
  - Class probabilities (`predict_proba`)
- Ideal for probabilistic sentiment interpretation

#### Support Vector Machine (LinearSVC)
- Optimized for large, sparse datasets
- Uses a **linear kernel**
- Outputs:
  - Class predictions
  - Decision score (distance from hyperplane)
- Faster and memory-efficient for text classification

---

## Evaluation & Visualization

For both models:

- **Classification Report**
  - Precision
  - Recall
  - F1-score
  - Accuracy
- **Confusion Matrix**
  - Saved as PNG images:
    - `confusion_matrix.png` (Logistic Regression)
    - `svm_confusion_matrix.png` (SVM)

Matplotlib is configured with a non-interactive backend (`Agg`) to ensure compatibility in headless environments.

---

## Dataset

### IMDB Movie Reviews Dataset

- **Provided by NLTK**

    - 2000 total reviews:
        - 1000 Positive
        - 1000 Negative

---

## Usage
**Logistic Regression Model** <br>
**python main.py**

**Support Vector Machine Model** <br>
**python SVM.py**

---

## Interactive Testing

Both models allow sentiment prediction on **custom user input**.

Example input:
```text
"This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!"