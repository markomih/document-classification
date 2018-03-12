from benchmark import Benchmark
from data_provider import NewsgroupsDataProvider, ReutersDataProvider
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from feature_extractor import FeatureExtractor
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('iris_rbf_svm')

ex.add_config({
    'max_df': 0.50,  # ignore terms that appear in more than 50% of the documents
    'min_df': 0.01,  # ignore terms that appear in less than 1% of the documents
    'max_features': 1000,
    'lowercase': True,
    'stop_words': 'english',
    'analyzer': 'word',
    'strip_accents': 'unicode'
})
ex.add_config({
    'use_idf': True,
    'sublinear_tf': True,
    'norm': None
})


@ex.named_config
def variant1():
    max_df = .40


# @ex.config
# def cfg():
#     count_vector = {
#         'max_df': 0.50,  # ignore terms that appear in more than 50% of the documents
#         'min_df': 0.01,  # ignore terms that appear in less than 1% of the documents
#         'max_features':1000,
#         'lowercase': True,
#         'stop_words':'english',
#         'analyzer': 'word',
#         'strip_accents':'unicode'
#     }

#

@ex.automain
def run(max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents,
        use_idf, sublinear_tf, norm):
    print('max_df=', max_df)

    count_vector = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, lowercase=lowercase,
                                   stop_words=stop_words, analyzer=analyzer, strip_accents=strip_accents)
    tfidf_transformer = TfidfTransformer(use_idf=use_idf, sublinear_tf=sublinear_tf, norm=norm)
    feature_extractor = FeatureExtractor(PorterStemmer(), WordNetLemmatizer(), count_vector, tfidf_transformer)
    data_provider = ReutersDataProvider()
    benchmark = Benchmark(data_provider, feature_extractor)
    benchmark.evaluate()


ex.observers.append(MongoObserver.create(url='my.server.org:27017', db_name='MY_DB'))  # TODO connect it with mlab

# if __name__ == '__main__':
#     ex.run(named_configs=['variant1'])
ex.run(named_configs=['variant1'])
