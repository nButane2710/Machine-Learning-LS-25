import pandas as pd
import numpy as np
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')

#task-0
df = pd.read_csv("q1_spam.csv", encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['Label', 'Message']

#task-1
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

df['tokens'] = df['Message'].apply(preprocess)

#task-2
print("Loading Word2Vec model (this may take time)...")
w2v_model = api.load("word2vec-google-news-300")

#task-3
def vectorize(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

df['vector'] = df['tokens'].apply(lambda tokens: vectorize(tokens, w2v_model))

#task-4
X = np.stack(df['vector'].values)
y = LabelEncoder().fit_transform(df['Label'])  # ham -> 0, spam -> 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#task-5
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}")

#task-6
def predict_message_class(model, w2v_model, message):
    tokens = preprocess(message)
    vector = vectorize(tokens, w2v_model).reshape(1, -1)
    pred = model.predict(vector)[0]
    return 'spam' if pred == 1 else 'ham'

test_msg = "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now."
print(f"Predicted class: {predict_message_class(clf, w2v_model, test_msg)}")
