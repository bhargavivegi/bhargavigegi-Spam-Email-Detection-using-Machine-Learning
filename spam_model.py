import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ✅ Read dataset
try:
    df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"], encoding="latin-1")
except FileNotFoundError:
    df = pd.read_csv("data/SMSSpamCollection.txt", sep="\t", names=["label", "message"], encoding="latin-1")

# ✅ Convert labels: ham → 0, spam → 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ✅ Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# ✅ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# ✅ Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_features, y_train)

# ✅ Predict on test data
y_pred = model.predict(X_test_features)

# ✅ Print evaluation results
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))
