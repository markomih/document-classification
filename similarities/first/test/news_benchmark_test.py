from benchmark import Benchmark
from data_provider import NewsgroupsDataProvider
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from feature_extractor import FeatureExtractor

count_vector = CountVectorizer(min_df=3, max_df=0.90, max_features=100, lowercase=True, stop_words='english')
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
feature_extractor = FeatureExtractor(PorterStemmer(), WordNetLemmatizer(), count_vector, tfidf_transformer)
dataProvider = NewsgroupsDataProvider()
benchmark = Benchmark(dataProvider, feature_extractor)
print('evaluation started')
benchmark.evaluate()