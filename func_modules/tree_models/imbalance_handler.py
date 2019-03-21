from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



"""
注意点：
（1）采取适合于不平衡样本的评价指标，如kappa、confusion matrix、roc_auc
（2）cv数能够满足稀有样本的使用
"""


def kappa_coef(y_true, y_pred):
    """
    kappa系数进行一致性检验
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    mat_dim = len(confusion_mat)
    sample_num = confusion_mat.sum(axis=0).sum()
    true_sum = 0
    for i in range(mat_dim):
        true_sum += confusion_mat[i, i]
    confusion_mat.sum(0)
    dot_sum = np.dot(confusion_mat.sum(axis=0), confusion_mat.sum(axis=1))
    p0 = true_sum/sample_num
    pe = dot_sum/(sample_num * sample_num)
    k = (p0 - pe)/(1 - pe)
    return k


# 1. 方法一：调整不同类型样本在模型（代价函数）中权重
# score = make_scorer(f1_score)
# score_k = make_scorer(kappa_coef)
# rf = RandomForestClassifier(random_state=80, class_weight="balanced")
# params = {"max_depth": np.arange(1, 5), "n_estimators": range(10, 50)}
# rf_grid = GridSearchCV(rf, param_grid=params, scoring=score_k)
# rf_grid.fit(X_train, Y_train)
# estimator = rf_grid.best_estimator_
# Y_pred = estimator.predict(X_test)
# res = confusion_matrix(Y_test, Y_pred)
# print(res)


# 2. 方法二：SMOTE方法来进行上采样或下采样
# from collections import defaultdict
# from imblearn.over_sampling import SMOTE
#
#
# def label_count(y):
#     num_dict = defaultdict(int)
#     for i in y:
#         num_dict[str(i)] += 1
#     print(num_dict)
#
# label_count(Y_train)
#
# # 上采样，丰富稀有类
# model_smote = SMOTE()
# x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X_train, Y_train)
#
# label_count(y_smote_resampled)
#
#
# # 下采样，减小丰富类
# from imblearn.under_sampling import RandomUnderSampler
# model_RandomUnderSampler = RandomUnderSampler()
# x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(X_train, Y_train)
#
# label_count(y_RandomUnderSampler_resampled)


# 3. 方法三：多个学习器融合
# 需提前观察丰富类和稀有类的样本比例，选取合适的学习器数量以及每个学习器内的正、负样本数量
# 比如这里采用有放回采样的形式，每个学习器选取正、负样本各20个

from collections import defaultdict


class Resample_Ensemble_Estimator(object):

    def __init__(self):
        self.data_x = None
        self.data_y = None
        self.size = None
        self.y_dict = None
        self.y_len_dict = None
        self.y_labels = None
        self.data = None
        self.estimator_list = list()

    def load_data(self, data_x, data_y, size):
        self.data_x = data_x
        self.data_y = data_y
        self.size = size

        self.y_dict = defaultdict(list)      # 存储每个y_label下的样本index
        self.y_len_dict = defaultdict(int)      # 存储每个y_label下的样本长度

        self.y_labels = list(set(self.data_y))    # y_label的数量
        self.data = np.c_[self.data_x, self.data_y]

        for label in self.y_labels:
            label_list = self.data[self.data[:, -1] == label]
            self.y_dict[str(label)] = label_list
            self.y_len_dict[str(label)] = len(label_list)

        assert self.size <= sorted(self.y_len_dict.values())[0]

    def imbalanced_resample(self):
        resample_data = np.empty(shape=(self.size*len(self.y_labels), self.data.shape[1]))
        print(resample_data.shape)

        # 拼接新的数据集
        for index, label in enumerate(self.y_labels):
            data_temp = self.y_dict[str(label)]
            resample_data[index*self.size:(index+1)*self.size, :] = data_temp[np.random.choice(range(len(data_temp)),
                                                                                          size=self.size, replace=False)]
            print(resample_data)
        return resample_data[:, :-1], resample_data[:, -1]

    def ensemble_estimator(self, base_estimator, num_ensemble, **kwargs):
        estimator = base_estimator(**kwargs)
        if not num_ensemble % 2:
            num_ensemble += 1
        for i in range(num_ensemble):
            train_res_x, train_res_y = self.imbalanced_resample()
            estimator.fit(train_res_x, train_res_y)
            self.estimator_list.append(estimator)
        return self.estimator_list

    def predict(self, x_test):
        y_pred_list = []
        y_pred_ensemble = []
        for est in self.estimator_list:
            y_pred = est.predict(x_test)
            y_pred_list.append(y_pred)
        for i in np.array(y_pred_list).T:
            y_pred_ensemble.append(pd.value_counts(i).sort_values().index[-1])
        return np.array(y_pred_ensemble)


if __name__ == '__main__':

    iris = load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    features = iris["feature_names"]

    estimators = Resample_Ensemble_Estimator()
    estimators.load_data(X_train, Y_train, size=5)
    res = estimators.ensemble_estimator(RandomForestClassifier, 5)
    for item in res:
        tt = item.predict(X_test)
        print(tt)
        print("______")

    res2 = estimators.predict(X_test)
    print(res2)

    np.concatenate()



