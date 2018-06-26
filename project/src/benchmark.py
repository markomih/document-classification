from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.linalg import norm
import math

from similarity_measure import EuclideanMeasure, CosineMeasure, TsSsMeasure


class Benchmark:
    def __init__(self, features, log_file):
        self.features = features
        self.log_file = log_file

    def new_evaluate(self):
        eucl, cos, tsss = 0, 0, 0

        euclidean_measure = EuclideanMeasure(self.features.train_features)
        cosine_measure = CosineMeasure(self.features.train_features)
        tsss_measure = TsSsMeasure(self.features.train_features, euclidean_measure, cosine_measure)

        for i in range(len(self.features.test_features)):
            print(self.features.test_features[i])
            ed_argmin = euclidean_measure.argmin_distance(self.features.test_features[i])
            cos_argmax = cosine_measure.argmax_distance(self.features.test_features[i])
            tass_argmin = tsss_measure.argmin_distance(self.features.test_features[i])

            if self.features.train_ctg[ed_argmin] == self.features.test_ctg[i]: eucl += 1
            if self.features.train_ctg[cos_argmax] == self.features.test_ctg[i]: cos += 1
            if self.features.train_ctg[tass_argmin] == self.features.test_ctg[i]: tsss += 1
            # TODO add logging
        return eucl, cos, tsss, len(self.features.test_features)
    # def evaluate(self):
    #     eucl, cos_max, ts_ss_min = 0, 0, 0
    #     n_a = norm(self.features.train_features, axis=1)
    #     for i in range(len(self.features.test_features)):
    #         n_b = norm(self.features.test_features[i])
    #         a_b = self.features.train_features - self.features.test_features[i]
    #         d_ab = np.dot(self.features.train_features, self.features.test_features[i])
    #         n = n_a * n_b
    #         ed = norm(a_b, axis=1)
    #
    #         n[n == 0] = -1
    #         cosine = (d_ab / n)
    #
    #         theta = np.arccos(cosine) + math.radians(10)
    #         ta = (n_a * n_b * np.degrees(np.sin(theta))) / 2
    #         md = np.abs(n_a - n_b)
    #         ss = np.pi * np.power(md + ed, 2) * np.degrees(theta) / 360
    #         tass = ta * ss
    #         # tass = (n_a*n_b * np.sin(theta)) * np.power(md + ed, 2) * theta
    #
    #         if self.features.train_ctg[ed.argmin()] == self.features.test_ctg[i]: eucl += 1
    #         if self.features.train_ctg[cosine.argmax()] == self.features.test_ctg[i]: cos_max += 1
    #         if self.features.train_ctg[tass.argmin()] == self.features.test_ctg[i]: ts_ss_min += 1
    #
    #         if i % 10 == 0: self.log(i, len(self.features.test_features), cos_max, eucl, ts_ss_min)
    #
    #     self.log(len(self.features.test_features), len(self.features.test_features), cos_max, eucl, ts_ss_min, True)
    #     # return ' '.join([str(cos_max), str(eucl), str(ts_ss_min), str(len(test_features))])
    #     # TODO add logger for metrics

    def log(self, iteration, total_count, cos_max, eucl, ts_ss_min, save=False):
        s = "iteration " + str(iteration) + "\\" + str(total_count) + '\n'
        s = s + 'cos_max=' + str(cos_max) + '\n'
        s = s + 'eucl=' + str(eucl) + '\n'
        s = s + 'ts_ss_min=' + str(ts_ss_min) + '\n'
        print(s)
        if save:
            with open(self.log_file, 'w') as f:
                f.write(s)
