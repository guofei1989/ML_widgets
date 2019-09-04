"""
示例为基于RFE递归筛选的特征选择功能
"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# 载入数据
data = load_iris()
data_X = data["data"]
data_y = data["target"]


# RFE
# sk = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
# estimator = DecisionTreeClassifier()
# rfe = RFE(estimator=estimator, n_features_to_select=2)
# rfe.fit(data_X, data_y)
# print(rfe.n_features_)
# print(rfe.ranking_)
# print(rfe.support_)
# print(rfe.estimator_)

# StratifiedShuffleSplit测试
# sss = StratifiedShuffleSplit(n_splits=2, random_state=40)
# for ind1, ind2 in sss.split(data_X, data_y):
#     print(random.sample)
#     print("-----"*5)
#     print(ind1)
#     print(ind2)


A = np.array([1,2,3,4,5])
# np.random.seed(1)
# print(np.random.sample)
# np.random.shuffle(A)
# print(A)
# np.random.seed(1)
# print(np.random.sample)
# np.random.shuffle(A)
# print(A)
rng = np.random.RandomState(10)
for _ in range(5):
    sort_list = np.random.permutation(10)
    print("------"*5)
    print(sort_list)
