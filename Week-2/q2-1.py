import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]


def tokenize(doc):
    return doc.lower().split()

tokenized_corpus = [tokenize(doc) for doc in corpus]

vocab = sorted(set(word for doc in tokenized_corpus for word in doc))
vocab_index = {word: i for i, word in enumerate(vocab)}

#Manual Term Frequency (TF)
def compute_tf(doc):
    tf = [0] * len(vocab)
    for word in doc:
        tf[vocab_index[word]] += 1
    return tf

tf_matrix = [compute_tf(doc) for doc in tokenized_corpus]

#Manual Document Frequency (DF)
df = [0] * len(vocab)
for i in range(len(vocab)):
    df[i] = sum(1 for doc in tokenized_corpus if vocab[i] in doc)

#Manual Inverse Document Frequency (IDF)
def compute_idf(df, N):
    return [math.log((N / df_i), 10) if df_i else 0 for df_i in df]

idf = compute_idf(df, len(corpus))

manual_tfidf = []
for tf_doc in tf_matrix:
    tfidf_doc = [tf * idf_val for tf, idf_val in zip(tf_doc, idf)]
    manual_tfidf.append(tfidf_doc)

print("\nManual TF-IDF Scores:")
print("Vocabulary:", vocab)
for i, vec in enumerate(manual_tfidf):
    print(f"Doc {i+1}: {vec}")

#Using sklearn CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(corpus).toarray()
print("\nCountVectorizer Output:")
print(cv.get_feature_names_out())
for i, row in enumerate(cv_matrix):
    print(f"Doc {i+1}: {row}")

#Using sklearn TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus).toarray()
print("\nTfidfVectorizer Output:")
print(tfidf.get_feature_names_out())
for i, row in enumerate(tfidf_matrix):
    print(f"Doc {i+1}: {row}")
