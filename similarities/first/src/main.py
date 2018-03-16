from nltk import PorterStemmer, WordNetLemmatizer
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from benchmark import Benchmark
from data_provider import DataProvider
from feature_extractor import FeatureExtractor

# ex = Experiment('tsss')
ex = Experiment()

URL = r'mongodb://markomihajlovicfm:itisme1994@ds115124.mlab.com:15124/intsys'
ex.observers.append(MongoObserver.create(url=URL, db_name='intsys'))


@ex.automain
def run(max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents,
        use_idf, sublinear_tf, norm, data_provider):

    print(max_df, min_df, max_features, lowercase, stop_words, analyzer, strip_accents,
        use_idf, sublinear_tf, norm, data_provider)
    count_vector = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, lowercase=lowercase,
                                   stop_words=stop_words, analyzer=analyzer, strip_accents=strip_accents)
    tfidf_transformer = TfidfTransformer(use_idf=use_idf, sublinear_tf=sublinear_tf, norm=norm)
    feature_extractor = FeatureExtractor(PorterStemmer(), WordNetLemmatizer(), count_vector, tfidf_transformer)
    benchmark = Benchmark(DataProvider.get_data_provider(data_provider), feature_extractor, False)
    return benchmark.evaluate()


# ex.run(named_configs=['variant_reuters'])
