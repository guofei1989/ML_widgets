import numpy as np
import pandas as pd
from sklearn.utils import shuffle as util_shuffle
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

def make_circles(n_samples=100, shuffle=True, noise=None, random_state=None,
                 factor1=0.8, factor2=0.5):

        per_samples = int(n_samples/3)
        n_samples_in = per_samples
        n_samples_mid = per_samples
        n_samples_out = n_samples - n_samples_in - n_samples_mid
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_mid = np.linspace(0, 2 * np.pi, n_samples_mid, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)

        outer_circ_x = np.cos(linspace_out)
        outer_circ_y = np.sin(linspace_out)
        outer_data = np.vstack([outer_circ_x.T, outer_circ_y.T]).T
        mid_circ_x = np.cos(linspace_mid) * factor1
        mid_circ_y = np.sin(linspace_mid) * factor1
        mid_data = np.vstack([mid_circ_x.T, mid_circ_y.T]).T
        inner_circ_x = np.cos(linspace_in) * factor2
        inner_circ_y = np.sin(linspace_in) * factor2
        inner_data = np.vstack([inner_circ_x.T, inner_circ_y.T]).T

        X = np.vstack([outer_data, mid_data, inner_data])

        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                       np.ones(n_samples_mid, dtype=np.intp),
                       np.ones(n_samples_in, dtype=np.intp)*2])

        if shuffle:
            X, y = util_shuffle(X, y, random_state=random_state)

        return X, y


if __name__ == '__main__':
    X, y = make_circles(300, shuffle=False)

    # print(y)
    #
    # print(X.shape)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    outer, mid, inner = 0, 1, 2
    labels = np.full(300, -1.)
    labels[0] = outer
    labels[150] = mid
    labels[-1] = inner

    # plt.scatter(X[:, 0], X[:, 1], c=labels)
    # plt.show()

    # label_spread = LabelPropagation()
    label_spread = LabelSpreading(kernel='knn', alpha=0.8)

    label_spread.fit(X, labels)
    labels_pred = label_spread.transduction_

    plt.scatter(X[:, 0], X[:, 1], c=labels_pred)
    plt.show()
