from abc import ABCMeta, abstractmethod
from numpy.linalg import norm
import numpy as np


class SimilarityMeasure(metaclass=ABCMeta):
    def __init__(self, train_features):
        self.train_features = train_features
        self.test_features = None
        self.distance = None

    def set_test_features(self, test_features):
        self.test_features = test_features

    @abstractmethod
    def get_distance(self, test_feature_vector):
        pass

    def argmax_distance(self, test_feature_vector):
        self.get_distance(test_feature_vector)
        return self.distance.argmax()

    def argmin_distance(self, test_feature_vector):
        self.get_distance(test_feature_vector)
        return self.distance.argmax()


class EuclideanMeasure(SimilarityMeasure):
    def get_distance(self, test_feature_vector):
        self.distance = norm(self.train_features - test_feature_vector, axis=1)
        return self.distance


class CosineMeasure(SimilarityMeasure):
    def __init__(self, train_features):
        super().__init__(train_features)
        self.train_features_norm = norm(self.train_features, axis=1)

    def get_distance(self, test_feature_vector):
        n = self.train_features_norm * norm(test_feature_vector)
        # n[n == 0] = -1 TODO fix this if nessary
        d_ab = np.dot(self.train_features, test_feature_vector)
        self.distance = (d_ab / n)

        return self.distance


class TsSsMeasure(SimilarityMeasure):
    def __init__(self, train_features, euclidean_measure, cosine_measure):
        super().__init__(train_features)
        self.euclidean_measure = euclidean_measure
        self.cosine_measure = cosine_measure

        self.train_features_norm = self.cosine_measure.train_features_norm
        self.cosine_measure = CosineMeasure(self.train_features)
        self.euclidean_measure = EuclideanMeasure(self.train_features)

    def get_distance(self, test_feature_vector):
        n_b = norm(test_feature_vector)

        cosine = self.cosine_measure.get_distance(test_feature_vector)
        ed = self.euclidean_measure.get_distance(test_feature_vector)

        theta = np.arccos(cosine) + np.math.radians(10)
        ts = (self.train_features_norm * n_b * np.degrees(np.sin(theta))) / 2
        md = np.abs(self.train_features_norm - n_b)
        ss = np.pi * np.power(md + ed, 2) * np.degrees(theta) / 360
        tass = ts * ss

        self.distance = tass

        return self.distance
