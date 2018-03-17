import numpy as np
from nltk import word_tokenize, re


class Features:
    def __init__(self, train_features, test_features, train_ctg, test_ctg):
        self.train_features = train_features
        self.test_features = test_features
        self.train_ctg = train_ctg
        self.test_ctg = test_ctg


class FeatureExtractor:
    def __init__(self, data, stemmer, lemmatizer, count_vector, tfidf_transformer):
        self.data = data
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.count_vector = count_vector
        self.tfidf_transformer = tfidf_transformer

        self.count_vector.tokenizer = self.text_tokenizer

    def text_tokenizer(self, text: str):
        min_length = 3

        text = re.sub('\S*@\S*\s?', '', text)  # remove emails
        text = re.sub(r'^https?://.*[\r\n]*', '', text, flags=re.MULTILINE)  # remove websites

        words = word_tokenize(text, 'english')
        words = list(filter(lambda word: len(word) >= min_length, words))

        # text = (list(map(lambda x: self.stemmer.stem(x), words)))
        tokens = (list(map(lambda x: self.lemmatizer.lemmatize(x), words)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))

        return filtered_tokens

    def extract_features(self, documents, fit=True):
        if fit:
            self.count_vector.fit(documents)
        vector = self.count_vector.transform(documents)
        tfidf_vector = self.tfidf_transformer.fit_transform(vector)

        return tfidf_vector.toarray()

    def get_features(self, drop_zero_vectors=True):
        print('extracting features')
        train_features = self.extract_features(self.data.train, fit=True)
        test_features = self.extract_features(self.data.test, fit=False)

        train_ctg = self.data.train_ctg
        if drop_zero_vectors:
            to_discard = np.any(train_features > .1, axis=1)
            train_ctg, train_features = self.data.train_ctg[to_discard], train_features[to_discard]
        print('features extracted')

        print('train_features.shape=', train_features.shape, 'test_features.shape=', test_features.shape)

        return Features(train_features, test_features, train_ctg, self.data.test_ctg)
