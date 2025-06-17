
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import string

# task-0, preprocessing
df = pd.read_csv("q5_reviews.csv")
df['Review'] = df['Review'].str.lower()
df['Review'] = df['Review'].apply(lambda x: re.sub(r'<.*?>', '', x))  
df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.punctuation))
df['Review'] = df['Review'].str.replace(r'\s+', ' ', regex=True)

# task-1
vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# task-2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# task-3
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# task-4
def text_preprocess_vectorize(texts, vectorizer):
    processed = [re.sub(r'<.*?>', '', t.lower()) for t in texts]
    processed = [t.translate(str.maketrans('', '', string.punctuation)) for t in processed]
    processed = [re.sub(r'\s+', ' ', t) for t in processed]
    return vectorizer.transform(processed)

if __name__ == "__main__":
    test_samples = ["Absolutely loved it", "Worst purchase ever"]
    vectors = text_preprocess_vectorize(test_samples, vectorizer)
    predictions = model.predict(vectors)
    for review, sentiment in zip(test_samples, predictions):
        print(f"Review: {review} -> Sentiment: {sentiment}")
