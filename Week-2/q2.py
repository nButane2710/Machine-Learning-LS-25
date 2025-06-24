import pandas as pd
import numpy as np
import re
import nltk
import gensim.downloader as api
import contractions
import emoji

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#task-0
df = pd.read_csv("q2_tweets.csv")[['airline_sentiment', 'text']]

#task-1
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    text = emoji.replace_emoji(text, replace='')
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

df['tokens'] = df['text'].apply(preprocess)

#task-2
print("Loading Word2Vec model...")
w2v_model = api.load("word2vec-google-news-300")

def vectorize(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df['vector'] = df['tokens'].apply(lambda tokens: vectorize(tokens, w2v_model))

#task-3
X = np.stack(df['vector'].values)
y = LabelEncoder().fit_transform(df['airline_sentiment'])  # 0=negative, 1=neutral, 2=positive

#task-4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#task-5
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.4f}")

#task-6
def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess(tweet)
    vec = vectorize(tokens, w2v_model).reshape(1, -1)
    pred = model.predict(vec)[0]
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[pred]

example_tweet = "I had a great flight experience with Delta!"
print(f"Predicted sentiment: {predict_tweet_sentiment(clf, w2v_model, example_tweet)}")
