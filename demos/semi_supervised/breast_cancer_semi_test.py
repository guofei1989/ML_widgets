import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt


data = pd.read_csv('breast-cancer-wisconsin_cleaned.csv')
data.to_csv()
#
# scaler = StandardScaler()

Y = data[['Class']]
X = data.drop('Class', axis=1)


# X_std = scaler.fit_transform(X)
# X_pd = pd.DataFrame(X_std, columns=X.columns)
# X = X_pd


from sklearn.model_selection import StratifiedShuffleSplit

testsize_list = []
train_socres_list = []
test_scores_list = []

def semi_shuffle_estimator(n_splits=10, test_size=0.6, seed=0, gamma=4, n_neighbors=6, max_iter=1000):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    i = 0
    testsize_list.append(test_size)
    train_scores = []
    test_scores =[]
    for label_index, unlabel_index in sss.split(X, Y):
        i += 1
        X_train = X.iloc[label_index]
        Y_train = Y.iloc[label_index]
        X_test = X.iloc[unlabel_index]
        Y_test = Y.iloc[unlabel_index]

        Y_unlabel = copy.deepcopy(Y_test)
        Y_unlabel['Class'] = -1

        X_new = pd.concat([X_train, X_test])
        Y_new = pd.concat([Y_train, Y_unlabel])


        shuffle_index = np.random.permutation(X.index)
        X_new_shuffle = X_new.take(shuffle_index)
        Y_new_shuffle = Y_new.take(shuffle_index)

        lp = LabelPropagation(gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter)
        lp.fit(X_new_shuffle, Y_new_shuffle.values.ravel())

        Y_predict_train = lp.predict(X_train)
        Y_predict_test = lp.predict(X_test)
        train_scores.append(accuracy_score(Y_train, Y_predict_train))
        test_scores.append(accuracy_score(Y_test, Y_predict_test))
        # print("-------Cross_validation epoch {}--------".format(i))
        # print("The accuracy in train set:", accuracy_score(Y_train, Y_predict_train))
        # print("The accuracy in test set:", accuracy_score(Y_test, Y_predict_test))
    mean_train_score = np.array(train_scores).mean()
    mean_test_score = np.array(test_scores).mean()
    print("For test size {}, the mean accuracy in train set is {}".format(test_size, mean_train_score))
    print("For test size {}, the mean accuracy in test set is {}".format(test_size, mean_test_score))
    train_socres_list.append(mean_train_score)
    test_scores_list.append(mean_test_score)


if __name__ == '__main__':

    for ratio in np.linspace(0.1, 0.94, 20):
        semi_shuffle_estimator(test_size=ratio)
    plt.plot(testsize_list, train_socres_list)
    plt.plot(testsize_list, test_scores_list)
    plt.show()

