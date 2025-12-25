import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.preprocess import clean_text

# Load dataset
df = pd.read_csv("data/tickets.csv")

# Combine Subject + Description
df["text"] = df["Subject"] + " " + df["Description"]
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ticket_classifier.pkl")

print("\n✅ Model trained and saved successfully!")
