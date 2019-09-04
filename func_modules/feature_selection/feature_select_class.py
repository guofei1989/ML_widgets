"""
构造一个用于特征选择的类，其包括方差检验、卡方检验、循环特征选取等方法

"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


class FeatureSelector(object):

    def __init__(self, select_feature_num):
        self.select_feature_num = select_feature_num
        self.X = None
        self.X_std = None
        self.y = None
        self.feature_names = None

    def read_data(self, path=None, target=None, x=None, y=None, feature=None):
        """可以从csv或xlsx中进行读取，也可以指定为DataFrame格式"""
        if path:
            if re.search(r'.csv', path):
                data = pd.read_csv(path)
                self.y = data[target]
                self.X = data.drop(target, axis=1)
                self.feature_names = list(self.X.columns)
            elif re.search(r'.xlsx', path):
                data = pd.read_excel(path)
                self.y = data[target]
                self.X = data.drop(target, axis=1)
                self.feature_names = list(self.X.columns)
            else:
                raise FileNotFoundError

        else:
            if isinstance(self.X, pd.DataFrame):
                self.X = x
                self.y = y
                self.feature_names = self.X.columns
            else:
                self.feature_names = list(feature)
                self.X = pd.DataFrame(x, columns=self.feature_names)
                self.y = y

    def variance_threshold(self, threshold=0):
        scaler = StandardScaler()
        self.X_std = scaler.fit_transform(self.X)
        # 去除方差小的特征
        var_sel = VarianceThreshold(threshold=threshold)
        var_sel.fit_transform(self.X_std)
        feature_var = list(var_sel.variances_)
        features = dict(zip(self.feature_names, feature_var))
        idx = var_sel.get_support()

        # topK的特征
        features_top = list(dict(sorted(features.items(), key=lambda d: d[1], reverse=True)))[:self.select_feature_num]
        # 超过方差threshold的特征
        feature_threshold = []
        for feature, flag in zip(feature_names, idx):
            if flag:
                feature_threshold.append(feature)
        return set(features_top) & set(feature_threshold)

    def select_k_best(self):
        # 单变量特征选取，注意卡方验证只适用于离散数据，而对于连续数值，需要用f_regression等
        sel = SelectKBest(chi2, k=self.select_feature_num)
        sel.fit(self.X, self.y)
        feature_var = list(sel.scores_)
        features = dict(zip(self.feature_names, feature_var))
        # print(features)
        features = list(dict(sorted(features.items(), key=lambda d: d[1], reverse=True)).keys())[:self.select_feature_num]
        return set(features)

    def rfe_select(self):
        # RFE循环特征选取
        svc = LinearSVC()    # 用线性核SVC也可以用其它线性分类器，若对于回归问题需要采用回归器
        rfe = RFE(estimator=svc, n_features_to_select=self.select_feature_num)
        rfe.fit_transform(self.X_std, self.y)
        features = dict(zip(self.feature_names, rfe.ranking_))
        # 或者可以通过 rfe.get_support()直接返回选择后的特征
        # print(features)
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[:self.select_feature_num]
        return features

    def tree_select(self):
        # 树模型嵌入式特征选取
        clf = ExtraTreeClassifier(max_depth=7)
        clf.fit(self.X, self.y.ravel())
        feature_var = list(clf.feature_importances_)
        features = dict(zip(self.feature_names, feature_var))
        # print(features)
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[:self.select_feature_num]
        return set(features)

    def return_feature_set(self, variance_threshold=False, select_k_best=False, rfe_select=False, tree_select=False):
        names = set([])
        assert isinstance(variance_threshold, bool)
        assert isinstance(select_k_best, bool)
        assert isinstance(rfe_select, bool)
        assert isinstance(tree_select, bool)

        if variance_threshold:
            name1 = self.variance_threshold()
            if not names:
                names = names.union(name1)
            else:
                names = names.intersection(name1)

        if select_k_best:
            name2 = self.select_k_best()
            if not names:
                names = names.union(name2)
            else:
                names = names.intersection(name2)

        if rfe_select:
            name3 = self.rfe_select()
            if not names:
                names = names.union(name3)
            else:
                names = names.intersection(name3)

        if tree_select:
            name4 = self.tree_select()
            if not names:
                names = names.union(name4)
            else:
                names = names.intersection(name4)

        return names


if __name__ == '__main__':
    selection = FeatureSelector(3)
    # data = load_boston()
    data2 = load_iris()
    data2_X = data2["data"]
    data2_y = data2["target"]
    feature_names = data2["feature_names"]

    selection.read_data(x=data2_X, y=data2_y, feature=feature_names)

    # features2 = selection.return_feature_set(variance_threshold=False, select_k_best=True, rfe_select=True, tree_select=True)
    # print(features2)
    print(selection.variance_threshold())
 