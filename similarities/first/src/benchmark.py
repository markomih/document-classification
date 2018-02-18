import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from data_provider import ReutersDataProvider
from vector_similarity import *


class Benchmark:
    def __init__(self):
        self.data_provider = ReutersDataProvider()
        self.cls = KNeighborsClassifier(1)

    def run(self):
        train_X, train_Y, test_X, test_Y = self.data_provider.get_all_features()

        count_cos_max = 0
        count_eucl_min = 0
        count_ts_ss_min = 0

        for i in range(len(test_X)):
            index = np.array([Cosine(test_X[i], train_X[j]) for j in range(len(train_X))]).argmax()
            if train_Y[index] == test_Y[i]: count_cos_max += 1

            index = np.array([Euclidean(test_X[i], train_X[j]) for j in range(len(train_X))]).argmin()
            if train_Y[index] == test_Y[i]: count_eucl_min += 1

            index = np.array([TS_SS(test_X[i], train_X[j]) for j in range(len(train_X))]).argmin()
            if train_Y[index] == test_Y[i]: count_ts_ss_min += 1

            if i % 10 == 0:
                print("iteration ", i, "\\", len(test_X))

        print('count_cos_max=', count_cos_max)
        print('count_eucl_min=', count_eucl_min)
        print('count_ts_ss=_min', count_ts_ss_min)
