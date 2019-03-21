import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def generate_samples():
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)
    return X, y


if __name__ == '__main__':
    X, y = generate_samples()
    estimator = KMeans()
