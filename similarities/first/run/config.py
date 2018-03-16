import os


class CountVectorizerConfiguration:
    def __init__(self, max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.analyzer = analyzer
        self.strip_accents = strip_accents

    def get_params(self):
        return [
            'max_df=' + str(self.max_df),
            'min_df=' + str(self.min_df),
            'max_features=' + str(self.max_features),
            'lowercase=' + str(self.lowercase),
            'stop_words=' + str(self.stop_words),
            'analyzer=' + str(self.analyzer),
            'strip_accents=' + str(self.strip_accents),
        ]

    def __str__(self):
        return ' '.join([
            'max_df=' + str(self.max_df),
            'min_df=' + str(self.min_df),
            'max_features=' + str(self.max_features),
            'lowercase=' + str(self.lowercase),
            'stop_words=' + str(self.stop_words),
            'analyzer=' + str(self.analyzer),
            'strip_accents=' + str(self.strip_accents),
        ])


class TfidfTransformerConfiguration:
    def __init__(self, use_idf, sublinear_tf, norm):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm

    def get_params(self):
        return [
            'use_idf=' + str(self.use_idf),
            'sublinear_tf=' + str(self.sublinear_tf),
            'norm=' + str(self.norm),
        ]

    def __str__(self):
        return ' '.join([
            'use_idf=' + str(self.use_idf),
            'sublinear_tf=' + str(self.sublinear_tf),
            'norm=' + str(self.norm),
        ])


class Configuration:
    def __init__(self, count_configuration, tfidf_configuration, data_provider: str):
        self.count_configuration = count_configuration
        self.tfidf_configuration = tfidf_configuration
        self.data_provider = data_provider

    def get_params(self):
        return self.count_configuration.get_params() + self.tfidf_configuration.get_params() + ['data_provider=' + str(self.data_provider)]

    def __str__(self):
        return ' '.join([
            str(self.count_configuration),
            str(self.tfidf_configuration),
            'data_provider=' + str(self.data_provider),
        ])
