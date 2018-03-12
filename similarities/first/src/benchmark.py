import numpy as np
from numpy.linalg import norm
import math


class Benchmark:
    def __init__(self, data_provider, feature_extractor):
        self.data_provider = data_provider
        self.feature_extractor = feature_extractor

    def evaluate(self):
        print('acquiring data')
        train, train_ctg, test, test_ctg = self.data_provider.get_documents()
        # train, train_ctg, test, test_ctg = train[:500], train_ctg[:500], test[:100], test_ctg[:100]
        print('data acquired')

        print('extracting features')
        train_features = self.feature_extractor.extract_features(train, fit=True)
        test_features = self.feature_extractor.extract_features(test, fit=False)

        to_discard = np.any(train_features > .1, axis=1)
        train_ctg, train_features = train_ctg[to_discard], train_features[to_discard]
        print('features extracted')

        print('number of train documents=', train_ctg.size, 'number of test documents=', train_ctg.size)
        print('train_features.shape=', train_features.shape, 'test_features.shape=', test_features.shape)

        eucl, cos_max, ts_ss_min = 0, 0, 0
        n_a = norm(train_features, axis=1)
        for i in range(len(test_features)):
            n_b = norm(test_features[i])
            a_b = train_features - test_features[i]
            d_ab = np.dot(train_features, test_features[i])
            n = n_a * n_b
            ed = norm(a_b, axis=1)

            n[n == 0] = -1
            cosine = (d_ab / n)

            theta = np.arccos(cosine) + math.radians(10)
            ta = (n_a * n_b * np.degrees(np.sin(theta))) / 2
            md = np.abs(n_a - n_b)
            ss = np.pi * np.power(md + ed, 2) * np.degrees(theta) / 360
            tass = ta * ss
            # tass = (n_a*n_b * np.sin(theta)) * np.power(md + ed, 2) * theta

            if train_ctg[ed.argmin()] == test_ctg[i]: eucl += 1
            if train_ctg[cosine.argmax()] == test_ctg[i]: cos_max += 1
            if train_ctg[tass.argmin()] == test_ctg[i]: ts_ss_min += 1

            if i % 10 == 0: self.log(i, len(test_features), cos_max, eucl, ts_ss_min)

        self.log(len(test_features), len(test_features), cos_max, eucl, ts_ss_min, True)

    @staticmethod
    def log(iteration, total_count, cos_max, eucl, ts_ss_min, save=False):
        s = "iteration " + str(iteration) + "\\" + str(total_count) + '\n'
        s = s + 'cos_max=' + str(cos_max) + '\n'
        s = s + 'eucl=' + str(eucl) + '\n'
        s = s + 'ts_ss_min=' + str(ts_ss_min) + '\n'
        print(s)
        if save:
            with open('log_norm_None.txt', 'w') as f:
                f.write(s)
