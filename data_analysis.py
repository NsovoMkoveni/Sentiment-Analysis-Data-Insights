import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD THE DATASET
# ==========================================

data = pd.read_csv("data/reviews.csv")


# ==========================================
# 2. BASIC DATA INFORMATION
# ==========================================

print("=" * 60)
print("SENTIMENT ANALYSIS - DATA INSIGHTS")
print("=" * 60)

print(f"\nTotal reviews: {len(data)}")

print("\nDataset columns:")
print(list(data.columns))


# ==========================================
# 3. SENTIMENT COUNTS
# ==========================================

sentiment_counts = data["sentiment"].value_counts()

print("\nSentiment Counts:")
print("-" * 60)

for sentiment, count in sentiment_counts.items():
    print(f"{sentiment.capitalize()}: {count}")


# ==========================================
# 4. SENTIMENT PERCENTAGES
# ==========================================

sentiment_percentages = (
    data["sentiment"].value_counts(normalize=True) * 100
)

print("\nSentiment Percentages:")
print("-" * 60)

for sentiment, percentage in sentiment_percentages.items():
    print(f"{sentiment.capitalize()}: {percentage:.2f}%")


# ==========================================
# 5. MOST COMMON SENTIMENT
# ==========================================

most_common_sentiment = sentiment_counts.idxmax()

print("\nMost Common Sentiment:")
print("-" * 60)
print(most_common_sentiment.capitalize())


# ==========================================
# 6. AI-ASSISTED DATA INSIGHT
# ==========================================

print("\nAI-Assisted Data Insight:")
print("-" * 60)

if most_common_sentiment == "positive":
    print(
        "The dataset contains mostly positive reviews, "
        "which suggests that customers generally had "
        "a favourable experience."
    )

elif most_common_sentiment == "negative":
    print(
        "The dataset contains mostly negative reviews, "
        "which suggests that there may be important "
        "customer satisfaction issues that need attention."
    )

else:
    print(
        "The dataset contains mostly neutral reviews, "
        "which suggests that customer experiences are "
        "generally mixed or average."
    )


# ==========================================
# 7. CREATE VISUALIZATION FOLDER
# ==========================================

import os

os.makedirs("visualizations", exist_ok=True)


# ==========================================
# 8. SENTIMENT DISTRIBUTION BAR CHART
# ==========================================

plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("visualizations/sentiment_distribution.png")

plt.close()


# ==========================================
# 9. SENTIMENT PERCENTAGE PIE CHART
# ==========================================

plt.figure(figsize=(7, 7))

sentiment_percentages.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sentiment Percentage Distribution")
plt.ylabel("")

plt.tight_layout()

plt.savefig("visualizations/sentiment_percentages.png")

plt.close()


# ==========================================
# 10. COMPLETION MESSAGE
# ==========================================

print("\nVisualizations created successfully!")

print("\nFiles saved:")
print("visualizations/sentiment_distribution.png")
print("visualizations/sentiment_percentages.png")

print("\n" + "=" * 60)
print("DATA ANALYSIS COMPLETE")
print("=" * 60)