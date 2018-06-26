from abc import abstractmethod, ABCMeta

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from benchmark import Benchmark
from data_provider import DataProvider
from feature_extractor import FeatureExtractor


class Configuration(metaclass=ABCMeta):
    @abstractmethod
    def get_params(self): pass

    @abstractmethod
    def __str__(self): pass


class CountVectorizerConfiguration(Configuration):
    def __init__(self, max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.analyzer = analyzer
        self.strip_accents = strip_accents

        self.count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features,
                                                lowercase=lowercase,
                                                stop_words=stop_words, analyzer=analyzer, strip_accents=strip_accents)

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
        return ' '.join(self.get_params())


class TfidfTransformerConfiguration(Configuration):
    def __init__(self, use_idf, sublinear_tf, norm):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm

        self.tfidf_transformer = TfidfTransformer(use_idf=use_idf, sublinear_tf=sublinear_tf, norm=norm)

    def get_params(self):
        return [
            'use_idf=' + str(self.use_idf),
            'sublinear_tf=' + str(self.sublinear_tf),
            'norm=' + str(self.norm),
        ]

    def __str__(self):
        return ' '.join(self.get_params())


class DataConfiguration(Configuration):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        data_provider = DataProvider.get_data_provider(dataset_name)
        self.data = data_provider.get_data()

    def get_params(self):
        return ['dataset=' + str(self.dataset_name)]

    def __str__(self):
        return ' '.join(self.get_params())


class FeaturesConfiguration(Configuration):
    def __init__(self, min_length, drop_zero_vectors, count_configuration, tfidf_configuration, data_configuration):
        self.min_length = min_length
        self.drop_zero_vectors = drop_zero_vectors
        self.count_configuration = count_configuration
        self.tfidf_configuration = tfidf_configuration
        self.data_configuration = data_configuration

        feature_extractor = FeatureExtractor(min_length, drop_zero_vectors, data_configuration.data,
                                             count_configuration.count_vectorizer,
                                             tfidf_configuration.tfidf_transformer)

        self.features = feature_extractor.get_features()
        self.count_configuration.max_features = self.features.train_features.shape[1]

    def __str__(self):
        return ' '.join(self.get_params())

    def get_params(self):
        tmp = [
            'min_length=' + str(self.min_length),
            'drop_zero_vectors=' + str(self.drop_zero_vectors),
        ]
        pars = self.count_configuration.get_params() + self.tfidf_configuration.get_params() + self.data_configuration.get_params()
        return tmp + pars


class BenchmarkConfiguration(Configuration):
    def __init__(self, feature_configuration):
        self.feature_configuration = feature_configuration

        self.benchmark = Benchmark(feature_configuration.features, 'mkr.txt')

    def get_params(self):
        return self.feature_configuration.get_params()

    def __str__(self):
        return ' '.join(self.get_params())

    def evaluation(self):
        eucl, cos, tsss, out_of = self.benchmark.new_evaluate()  # TODO add yield

        to_save = self.get_params() + ['eucl='+str(eucl), 'cos='+str(cos), 'tsss='+str(tsss), 'out_of='+str(out_of)]
        with open('log.txt', 'a') as f:
            s = ' '.join(to_save) + '\n'
            f.write(s)
