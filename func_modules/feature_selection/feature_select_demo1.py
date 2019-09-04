"""
特征选择的三种方法：
1. 过滤性
选择与目标变量相关性较强的特征。缺点：忽略了特征之间的关联性。

2. 包裹型
基于模型得到的特征重要性逐步剔除特征（贪心算法）。

3. 嵌入型
利用模型提取特征，一般基于线性模型与正则化（正则化取L1）,取权重非0的特征。

"""

from sklearn.datasets import load_iris, load_boston
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector(object):

    def __init__(self):
        self.X = None
        self.y = None

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def chi_selector(self, k=2):
        """过滤型方法：根据卡方验证的方法，选出最相关的特征"""
        return SelectKBest(chi2, k=k).fit_transform(self.X, self.y)

    def rfe_selector(self, num=3):
        lr = LinearRegression()    # 最好选取稳定性好一点的基学习器，
        rfe = RFE(lr, n_features_to_select=num)
        rfe.fit(self.X, self.y)
        return rfe

    def embedding_selector(self):
        lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(self.X, self.y)
        model = SelectFromModel(lsvc, prefit=True)
        return model.transform(self.X)
        # 可以调用get_support()方法，得到相关的特征
        # return model.get_support()     # 返回的是一个布尔值列表，表明每个维度是否为重要特征


if __name__ == '__main__':

    selector = FeatureSelector()

# 1. 卡方验证示例：

    # data = load_iris()
    # print(data.data.shape)
    # selector.load_data(data.data, data.target)
    # res = selector.chi_selector(k=2)
    # print(res.shape)

# 2. 包裹型示例：
#     boston = load_boston()
#     X = boston["data"]
#     Y = boston["target"]
#     names = boston["feature_names"]
#     print(len(names))
#     selector.load_data(X, Y)
#     rfe2 = selector.rfe_selector()
#     print(rfe2.ranking_)


# 3. 嵌入型示例：
    iris=load_iris()
    X,y=iris.data,iris.target
    selector.load_data(X, y)
    print(X.shape)
    X_new = selector.embedding_selector()
    print(X_new.shape)