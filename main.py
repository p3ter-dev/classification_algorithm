import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the IMDB movie reviews dataset
reviews = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        reviews.append(movie_reviews.raw(fileid))
        labels.append(0 if category == 'neg' else 1)  # 0 for negative, 1 for positive

# Preprocessing function
def preprocess(text):
    # Remove handles (e.g., @username) and URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [t.lower() for t in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]  # Keep only alphabetic words
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    # Join back into string
    return ' '.join(tokens)

# Preprocess all reviews
processed_reviews = [preprocess(r) for r in reviews]

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(processed_reviews)
y = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Negative", "Positive"],
    cmap="Blues"
)

disp.figure_.suptitle("Logistic Regression - Confusion Matrix (for IMDB)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Interactive Model Test

# Define the input review
my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'

print("\nTesting New Review")
print(f"Original Review: {my_tweet}")

# Preprocess the review (using your existing function)
# preprocess returns a single string
processed_tweet = preprocess(my_tweet) 
print(f"Processed Tokens: {processed_tweet}")

# Vectorize the processed review
# The model expects the input to be in the same format (TF-IDF vector) as the training data.
X_new = vectorizer.transform([processed_tweet])

# Predict the class (0 or 1)
# model.predict gives the class (0 or 1)
y_hat_class = model.predict(X_new)[0]

# Predict the probability (for detailed sentiment)
# model.predict_proba gives the probability of each class
y_hat_proba = model.predict_proba(X_new)[0] 
positive_proba = y_hat_proba[1] # Probability of being class 1 or Positive

# Print Results
print(f"Predicted Class (0=Neg, 1=Pos): {y_hat_class}")
print(f"Positive Sentiment Probability: {positive_proba:.4f}")

if positive_proba > 0.5:
    print('Prediction: Positive sentiment')
else: 
    print('Prediction: Negative sentiment')