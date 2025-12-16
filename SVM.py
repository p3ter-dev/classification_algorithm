import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# --- CHANGE: Import LinearSVC instead of LogisticRegression ---
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the IMDB movie reviews dataset (Reviews and Labels loading remains the same)
reviews = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        reviews.append(movie_reviews.raw(fileid))
        labels.append(0 if category == 'neg' else 1)  # 0 for negative, 1 for positive

# Preprocessing function (Remains the same)
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

# model training
# LinearSVC is highly efficient for large, sparse datasets like TF-IDF vectors.
model_svm = LinearSVC(random_state=42, dual='auto') 
model_svm.fit(X_train, y_train)

# Predict on test set
y_pred_svm = model_svm.predict(X_test)

# Print classification report
print("Classification Report (LinearSVC):")
print(classification_report(y_test, y_pred_svm))

# confusion martix (for SVM)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_svm,
    display_labels=["Negative", "Positive"],
    cmap="Blues"
)

disp.figure_.suptitle("LinearSVC (SVM) - Confusion Matrix (for IMDB)")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.close(disp.figure_)
print("\nConfusion matrix saved as 'svm_confusion_matrix.png'")

# Interactive Model Test (for SVM)

# Define the input review
my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'

print("\n--- Testing New Review with LinearSVC (SVM) ---")
print(f"Original Review: {my_tweet}")

# Preprocess the review
processed_tweet = preprocess(my_tweet) 
print(f"Processed Tokens: {processed_tweet}")

# Vectorize the processed review
X_new = vectorizer.transform([processed_tweet])

# Predict the class (0 or 1)
y_hat_class_svm = model_svm.predict(X_new)[0]

# Get the SVM decision score
# SVMs output a 'decision score' (distance from the hyperplane), not a probability.
# Scores > 0 mean positive, Scores < 0 mean negative.
decision_score = model_svm.decision_function(X_new)[0] 

# Print Results
print(f"Predicted Class (0=Neg, 1=Pos): {y_hat_class_svm}")
print(f"SVM Decision Score: {decision_score:.4f}")

if decision_score > 0:
    print('Prediction: Positive sentiment')
else: 
    print('Prediction: Negative sentiment')