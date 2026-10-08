from flask import Flask, render_template, request
import joblib


# ==========================================
# CREATE FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD TRAINED SENTIMENT MODEL
# ==========================================

model = joblib.load("models/sentiment_model.pkl")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    review = ""

    if request.method == "POST":

        review = request.form["review"]

        if review.strip():

            prediction = model.predict([review])[0]

            probabilities = model.predict_proba([review])[0]

            confidence = max(probabilities) * 100

            sentiment = prediction.capitalize()

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)