# 🤖 AI Sentiment Analyzer

## Customer Sentiment Analysis

An AI-powered web application that analyzes customer reviews and classifies them as **Positive, Negative, or Neutral** using machine learning.

The project was developed as part of an **AI Bootcamp** under the theme **AI for Data Analysis & Insights**.

---

## 🚀 Live Application

👉 [Launch AI Sentiment Analyzer](https://sentiment-analysis-data-insights.onrender.com)

---

## 📌 Project Overview

Customer feedback contains valuable information that can help organizations understand customer experiences.

This project uses a machine learning model to analyze written customer reviews and automatically determine their sentiment.

The application provides:

- 😊 Positive sentiment detection
- 😐 Neutral sentiment detection
- 😞 Negative sentiment detection
- 📊 Sentiment distribution visualization
- 🥧 Sentiment percentage visualization
- 🎯 Prediction confidence score
- 🌐 Interactive Flask web interface

---

## 🧠 Machine Learning Model

The sentiment classification model uses:

- **TF-IDF Vectorization**
- **Logistic Regression**
- **Scikit-learn**
- **Natural Language Processing (NLP)**

The model was trained using a dataset containing **316 customer reviews** across three sentiment categories:

| Sentiment | Reviews |
|-----------|---------|
| Positive | 105 |
| Neutral | 106 |
| Negative | 105 |

---

## 📊 Model Performance

The dataset was divided into training and testing data.

- Training reviews: **252**
- Testing reviews: **64**
- Test accuracy: **76.56%**

The model achieved an overall test accuracy of **76.56%**.

> Note: Model predictions are based on patterns learned from the training dataset. Some reviews may be classified incorrectly, particularly when wording contains mixed or ambiguous sentiment.

---

## 📈 Data Insights

The application includes visualizations showing:

### Sentiment Distribution

Displays the number of reviews belonging to each sentiment category.

### Sentiment Percentage Distribution

Shows the percentage breakdown of positive, neutral, and negative reviews.

The dataset is relatively balanced across the three sentiment categories, with neutral reviews being slightly more common.

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- HTML
- CSS
- Machine Learning
- Natural Language Processing

---

## 📂 Project Structure

```text
Sentiment-Analysis-Data-Insights/
│
├── app.py
├── data_analysis.py
├── sentiment_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── reviews.csv
│
├── models/
│   └── sentiment_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── sentiment_distribution.png
│   └── sentiment_percentages.png
│
└── visualizations/
    ├── sentiment_distribution.png
    └── sentiment_percentages.png