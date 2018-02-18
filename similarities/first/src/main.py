import numpy as np

from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

from vector_similarity import *


def get_documents(category):
    train = [field for field in reuters.fileids(categories=[category]) if field.startswith('train')]
    test = [field for field in reuters.fileids(categories=[category]) if field.startswith('test')]
    for field in reuters.fileids(categories=['cocoa']):
        print(field)

    train = [reuters.raw(t) for t in train]
    test = [reuters.raw(t) for t in test]

    return train, test


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ctgs = reuters.categories()

train_docs, test_docs = get_documents('cocoa')
train_docs_castor, test_docs_castor = get_documents('castor-oil')
train_docs_corn, test_docs_corn = get_documents('corn')

train_docs = test_docs + test_docs_castor + test_docs_corn
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_docs)
X = X.toarray()
print(X.shape)
cls = LogisticRegression(penalty='l1', n_jobs=-1)
cls.fit()
plt.plot(X[0], color='r')
plt.plot(X[1], color='b')
cos_sim = np.ones((len(X), len(X)))
ts_ss_sim = np.ones((len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        cos_sim[i, j] = cosine_similarity(X[i], X[j])
        ts_ss_sim[i, j] = TS_SS(X[i].tolist(), X[j].tolist())

        # if i != j:
        #     print(cosine_similarity(X[i], X[j]))
plt.plot(X[0], color='r')
plt.plot(X[1], color='g')
plt.plot(X[2], color='b')
plt.show()
