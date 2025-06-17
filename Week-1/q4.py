import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import string
import re

# task-0
df = pd.read_csv("q4_reviews.csv", sep=',', encoding='utf-8')

# pre-processing
df.dropna(subset=['Review', 'Sentiment'], inplace=True)
df['Sentiment'] = df['Sentiment'].str.strip()
df['Review'] = df['Review'].str.lower()
df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.punctuation))
df['Review'] = df['Review'].apply(lambda x: re.sub(r'<.*?>', '', x))  
df['Review'] = df['Review'].str.replace(r'\s+', ' ', regex=True)


# task-1
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# task-2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# task-3
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# task-4
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "count_vectorizer.pkl")

def predict_review_sentiment(model, vectorizer, review):
    vector = vectorizer.transform([review])
    prediction = model.predict(vector)
    return prediction[0]

if __name__ == "__main__":
    review = "Really good movie."
    loaded_model = joblib.load("naive_bayes_model.pkl")
    loaded_vectorizer = joblib.load("count_vectorizer.pkl")
    print("Predicted Sentiment:", predict_review_sentiment(loaded_model, loaded_vectorizer, review))