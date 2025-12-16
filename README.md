# Sentiment Analysis Classification Algorithm

This project implements a sentiment analysis classifier using Logistic Regression on IMDB movie reviews from the NLTK dataset.

## Features

- **Preprocessing**: Removes handles and URLs, tokenizes, converts to lowercase, removes stop words, and applies stemming.
- **Classification**: Uses Logistic Regression with TF-IDF vectorization.
- **Visualization**: Plots a confusion matrix for the classification results.

## Requirements

Install the required packages using:

```
pip install -r requirements.txt
```

## Usage

Run the main script:

```
python main.py
```

This will train the model, evaluate it on the test set, and display the confusion matrix plot.

## Dataset

The dataset used is the IMDB movie reviews from NLTK, which contains positive and negative movie reviews.