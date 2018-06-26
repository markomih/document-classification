from benchmark import Benchmark
from data_provider import NewsgroupsDataProvider, ReutersDataProvider
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from feature_extractor import FeatureExtractor

max_df = 0.50  # ignore terms that appear in more than 50% of the documents
min_df = 0.01  # ignore terms that appear in less than 1% of the documents

count_vector = CountVectorizer(min_df=min_df, max_df=max_df, max_features=1000, lowercase=True, stop_words='english',
                               analyzer='word', strip_accents='unicode')
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True, norm=None)
feature_extractor = FeatureExtractor(PorterStemmer(), count_vector, tfidf_transformer)
# dataProvider = NewsgroupsDataProvider()
dataProvider = ReutersDataProvider()
benchmark = Benchmark(dataProvider, feature_extractor)
print('evaluation started')
benchmark.evaluate()
