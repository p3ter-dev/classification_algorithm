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
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
# import gradio as gr # Removed gradio since it's not used below

# Download necessary NLTK data
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

# --- MODEL TRAINING: Switched to LinearSVC (SVM) ---
# LinearSVC is highly efficient for large, sparse datasets like TF-IDF vectors.
# Note: SVMs do not natively output probabilities, but we can enable decision_function.
model_svm = LinearSVC(random_state=42, dual='auto') 
model_svm.fit(X_train, y_train)

# Predict on test set
y_pred_svm = model_svm.predict(X_test)

# Print classification report
print("Classification Report (LinearSVC):")
print(classification_report(y_test, y_pred_svm))

# --- PLOT 1: CONFUSION MATRIX (for SVM) ---
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


# --- PLOT 2: PCA VISUALIZATION (for SVM) ---

# 1. Reduce Dimensions for Plotting
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test.toarray())

# 2. Identify Misclassified Points
misclassified_mask = (y_pred_svm != y_test)
correctly_classified_mask = ~misclassified_mask 

plt.figure(figsize=(10, 7))

# Plot CORRECTLY classified points (Faded)
plt.scatter(
    X_test_2d[correctly_classified_mask, 0], 
    X_test_2d[correctly_classified_mask, 1], 
    c=y_pred_svm[correctly_classified_mask], 
    cmap='bwr', alpha=0.2, s=50, label='Correctly Classified'
)

# Plot MISCLASSIFIED points (Highlighted)
plt.scatter(
    X_test_2d[misclassified_mask, 0], 
    X_test_2d[misclassified_mask, 1], 
    c='yellow', edgecolor='k', linewidth=1.5, s=100, label='Misclassified (Error)'
)

# Formatting
plt.title('LinearSVC (SVM) PCA Visualization: Highlighting Misclassified Reviews', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

yellow_patch = mpatches.Patch(color='yellow', label='Misclassified')
blue_patch = mpatches.Patch(color='blue', label='Predicted Negative (Correct)')
red_patch = mpatches.Patch(color='red', label='Predicted Positive (Correct)')

plt.legend(handles=[yellow_patch, blue_patch, red_patch], loc='best')

plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('svm_pca_misclassified_plot.png') 
print("Plot with highlighted errors saved as 'svm_pca_misclassified_plot.png'.")

# --- Interactive Model Test (for SVM) ---

# 1. Define the input review
my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'

print("\n--- Testing New Review with LinearSVC (SVM) ---")
print(f"Original Review: {my_tweet}")

# 2. Preprocess the review (same as before)
processed_tweet = preprocess(my_tweet) 
print(f"Processed Tokens: {processed_tweet}")

# 3. Vectorize the processed review
X_new = vectorizer.transform([processed_tweet])

# 4. Predict the class (0 or 1)
y_hat_class_svm = model_svm.predict(X_new)[0]

# 5. Get the SVM decision score (replaces probability)
# SVMs output a 'decision score' (distance from the hyperplane), not a probability.
# Scores > 0 mean positive, Scores < 0 mean negative.
decision_score = model_svm.decision_function(X_new)[0] 

# 6. Print Results
print(f"Predicted Class (0=Neg, 1=Pos): {y_hat_class_svm}")
print(f"SVM Decision Score: {decision_score:.4f}")

if decision_score > 0:
    print('Prediction: Positive sentiment')
else: 
    print('Prediction: Negative sentiment')