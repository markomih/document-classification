import numpy as np
from numpy.linalg import norm


class Benchmark:
    def __init__(self, data_provider, feature_extractor):
        self.data_provider = data_provider
        self.feature_extractor = feature_extractor

    def evaluate(self):
        train, train_ctg, test, test_ctg = self.data_provider.get_documents()
        print(len(train), len(test))
        # train, train_ctg, test, test_ctg = train, train_ctg, test, test_ctg
        print('data acquired')

        train_features = self.feature_extractor.extract_features(train, fit=True)
        test_features = self.feature_extractor.extract_features(test, fit=False)
        print('features extracted')

        print('number of train documents=', len(train), 'number of test documents=', len(test))
        print('train_features.shape=', train_features.shape, 'test_features.shape=', test_features.shape)

        cos_min, cos_max, eucl, ts_ss_min, ts_ss_max = 0, 0, 0, 0, 0
        # n_a = norm(train_features, axis=1)
        for i in range(len(test_features)):
            # n_b = norm(test_features[i])
            a_b = train_features - test_features[i]
            d_ab = np.dot(train_features, test_features[i])
            # n = n_a * n_b

            ed = norm(a_b, axis=1)
            # cosine = (d_ab / n)
            cosine = d_ab
            theta = np.arccos(cosine)
            # ta = (n * np.sin(np.radians(theta) + 10)) / 2
            ta = (np.sin(np.radians(theta) + 10)) / 2
            # ss = np.pi * np.power(np.abs(n_a - n_b) + ed, 2) * theta / 360
            ss = np.pi * np.power(ed, 2) * theta / 360
            tass = ta * ss

            if train_ctg[ed.argmin()] == test_ctg[i]: eucl += 1
            if train_ctg[cosine.argmin()] == test_ctg[i]: cos_min += 1
            if train_ctg[cosine.argmax()] == test_ctg[i]: cos_max += 1
            if train_ctg[tass.argmin()] == test_ctg[i]: ts_ss_min += 1
            if train_ctg[tass.argmax()] == test_ctg[i]: ts_ss_max += 1

            if i % 10 == 0: self.log(i, len(test_features), cos_min, cos_max, eucl, ts_ss_min, ts_ss_max)

        self.log(len(test_features), len(test_features), cos_min, cos_max, eucl, ts_ss_min, ts_ss_max, True)

    @staticmethod
    def log(iteration, total_count, cos_min, cos_max, eucl, ts_ss_min, ts_ss_max, save=False):
        s = "iteration " + str(iteration) + "\\" + str(total_count) + '\n'
        s = s + 'cos_min=' + str(cos_min) + '\n'
        s = s + 'cos_max=' + str(cos_max) + '\n'
        s = s + 'eucl=' + str(eucl) + '\n'
        s = s + 'ts_ss_min=' + str(ts_ss_min) + '\n'
        s = s + 'ts_ss_max=' + str(ts_ss_max) + '\n'
        print(s)
        if save:
            with open('log.txt', 'w') as f:
                f.write(s)
