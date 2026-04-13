import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("tickets.csv")

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df["cleaned"] = df["text"].apply(clean_text)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["category"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# PREDICTION & EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# GRAPH 1: BAR CHART
# -------------------------------
category_counts = df["category"].value_counts()

plt.figure()
category_counts.plot(kind='bar')
plt.title("Ticket Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# -------------------------------
# GRAPH 2: PIE CHART
# -------------------------------
plt.figure()
category_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title("Category Distribution")
plt.ylabel("")
plt.show()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_ticket(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    category = model.predict(vec)[0]

    # Priority logic
    if "not working" in text or "error" in text or "failed" in text:
        priority = "High"
    elif "slow" in text or "late" in text or "delay" in text:
        priority = "Medium"
    else:
        priority = "Low"

    return category, priority

# -------------------------------
# USER INPUT LOOP
# -------------------------------
while True:
    user_input = input("\nEnter your ticket (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Exiting program...")
        break

    category, priority = predict_ticket(user_input)

    print("Category:", category)
    print("Priority:", priority)
