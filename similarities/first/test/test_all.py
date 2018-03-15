from benchmark import Benchmark
from data_provider import NewsgroupsDataProvider, ReutersDataProvider, DataProvider
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from feature_extractor import FeatureExtractor
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('tsss')

URL = r'mongodb://markomihajlovicfm:itisme1994@ds115124.mlab.com:15124/intsys'
ex.observers.append(MongoObserver.create(url=URL, db_name='intsys'))

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


# ex.add_config({
#     'data_provider': 'reuters',
# })


@ex.named_config
def variant_reuters():
    data_provider = 'reuters'


@ex.named_config
def variant_newsgroup():
    data_provider = 'newsgroup'


@ex.automain
def run(max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents,
        use_idf, sublinear_tf, norm, data_provider):
    count_vector = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, lowercase=lowercase,
                                   stop_words=stop_words, analyzer=analyzer, strip_accents=strip_accents)
    tfidf_transformer = TfidfTransformer(use_idf=use_idf, sublinear_tf=sublinear_tf, norm=norm)
    feature_extractor = FeatureExtractor(PorterStemmer(), WordNetLemmatizer(), count_vector, tfidf_transformer)
    benchmark = Benchmark(DataProvider.get_data_provider('reuters'), feature_extractor, False)
    return benchmark.evaluate()


# ex.run(named_configs=['variant_reuters'])
