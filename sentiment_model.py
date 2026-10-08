import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ==========================================
# 1. LOAD THE DATASET
# ==========================================

data = pd.read_csv("data/reviews.csv")

X = data["review"]
y = data["sentiment"]


# ==========================================
# 2. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. CREATE THE MACHINE LEARNING MODEL
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=2.0
        )
    )
])


# ==========================================
# 4. TRAIN THE MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 5. EVALUATE THE MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 55)
print("SENTIMENT ANALYSIS MODEL")
print("=" * 55)

print(f"\nTraining reviews: {len(X_train)}")
print(f"Testing reviews: {len(X_test)}")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ==========================================
# 6. SAVE THE TRAINED MODEL
# ==========================================

joblib.dump(model, "models/sentiment_model.pkl")

print("Model saved successfully!")
print("Location: models/sentiment_model.pkl")


# ==========================================
# 7. TEST NEW REVIEWS
# ==========================================

test_reviews = [
    "I really love this product, it is amazing!",
    "The service was okay, nothing special.",
    "I am very disappointed with this terrible service.",
    "The staff were friendly and extremely helpful.",
    "The product stopped working and I am very unhappy."
]

print("\nNew Review Predictions:")
print("-" * 55)

for review in test_reviews:

    prediction = model.predict([review])[0]

    probabilities = model.predict_proba([review])[0]

    confidence = max(probabilities) * 100

    print(f"\nReview: {review}")
    print(f"Sentiment: {prediction}")
    print(f"Confidence: {confidence:.2f}%")