from nltk import word_tokenize, re


class FeatureExtractor:
    def __init__(self, stemmer, lemmatizer, count_vector, tfidf_transformer):
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
