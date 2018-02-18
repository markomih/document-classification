import numpy as np

from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer, WordNetLemmatizer, re


class ReutersDataProvider:
    porterStemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def __init__(self):
        self.categories = np.array(list(range(len(reuters.categories()))))

    @staticmethod
    def get_train_documents(category):
        train = [field for field in reuters.fileids(categories=[category]) if field.startswith('train')]
        train = [reuters.raw(t) for t in train]
        return train

    @staticmethod
    def get_test_documents(category):
        test = [field for field in reuters.fileids(categories=[category]) if field.startswith('test')]
        test = [reuters.raw(t) for t in test]
        return test

    @staticmethod
    def extract_features(documents):
        count_vect = CountVectorizer(tokenizer=ReutersDataProvider.text_tokenization, min_df=3, max_df=0.90,
                                     max_features=1000, lowercase=True, stop_words='english')

        x_train_counts = count_vect.fit_transform(documents)
        tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
        X = tfidf_transformer.fit_transform(x_train_counts)

        X = X.toarray()
        return X

    def get_all_features(self):
        train, test, train_label, test_label = [], [], [], []
        for l, category in enumerate(reuters.categories()[2:5]):
            tmp_train = self.get_train_documents(category)
            tmp_test = self.get_test_documents(category)

            train = train + tmp_train
            test = test + tmp_test
            train_label = train_label + [l] * len(tmp_train)
            test_label = test_label + [l] * len(tmp_test)

        train_X = self.extract_features(train)
        train_Y = np.array(train_label)

        test_X = self.extract_features(test)
        test_Y = np.array(test_label)
        return train_X, train_Y, test_X, test_Y

    @staticmethod
    def text_tokenization(text: str):
        min_length = 3
        words = word_tokenize(text, 'english')
        tokens = (list(map(lambda x: ReutersDataProvider.porterStemmer.stem(x), words)))
        tokens = (list(map(lambda x: ReutersDataProvider.lemmatizer.lemmatize(x), tokens)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))

        return filtered_tokens
