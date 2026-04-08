# =====================================
# 1️⃣ Import Libraries
# =====================================
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')

# =====================================
# 2️⃣ Load Dataset
# =====================================
df = pd.read_csv("tickets.csv")

print("\nDataset Loaded Successfully")
print("Columns:", df.columns)

# =====================================
# 3️⃣ Rename Columns
# =====================================
df = df.rename(columns={
    'Document': 'text',
    'Topic_group': 'category'
})

print("\nRenamed Columns:", df.columns)

# =====================================
# 4️⃣ Text Cleaning
# =====================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# =====================================
# 5️⃣ TF-IDF Vectorization (Improved with Bigrams)
# =====================================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])
y_category = df['category']

# =====================================
# 6️⃣ Train-Test Split (Category Model)
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_category, test_size=0.2, random_state=42
)

# =====================================
# 7️⃣ Train Category Classification Model
# =====================================
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train, y_train)

# =====================================
# 8️⃣ Evaluate Category Model
# =====================================
y_pred = category_model.predict(X_test)

print("\n===== Category Classification Results =====\n")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visual Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Category Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================
# 9️⃣ Create Priority Labels (Rule-Based for Training)
# =====================================
def assign_priority(text):
    text = text.lower()
    if "urgent" in text or "immediately" in text or "critical" in text:
        return "High"
    elif "error" in text or "issue" in text or "problem" in text:
        return "Medium"
    else:
        return "Low"

df['priority'] = df['clean_text'].apply(assign_priority)

y_priority = df['priority']

# =====================================
# 🔟 Train-Test Split (Priority Model)
# =====================================
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_train_p, y_train_p)

# =====================================
# 1️⃣1️⃣ Evaluate Priority Model
# =====================================
y_pred_p = priority_model.predict(X_test_p)

print("\n===== Priority Classification Results =====\n")
print("Accuracy:", round(accuracy_score(y_test_p, y_pred_p), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_p, y_pred_p))

# =====================================
# 1️⃣2️⃣ Test on New Ticket
# =====================================
new_ticket = ["System is not working and needs urgent fix"]
new_ticket_clean = clean_text(new_ticket[0])
new_vector = vectorizer.transform([new_ticket_clean])

predicted_category = category_model.predict(new_vector)
predicted_priority = priority_model.predict(new_vector)

print("\n===== New Ticket Test =====")
print("Ticket:", new_ticket[0])
print("Predicted Category:", predicted_category[0])
print("Predicted Priority:", predicted_priority[0])
