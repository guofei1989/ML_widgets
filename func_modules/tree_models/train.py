import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import Imputer,PolynomialFeatures,LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler,label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle
from sklearn.externals import joblib

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

# 读取数据
path = "./data/20190110/清洗空缺值.xlsx"
train = pd.read_excel(path)

# 因变量名称
target = 'IS_EAD'

deleteColumns = ['供体钾','供体体重','供体血糖']

no_x_columns = deleteColumns + [target]
x_columns = [x for x in train.columns if x not in [no_x_columns]]
X = train[x_columns]
y = train[target]

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print (rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

#n_estimators 参数调优
# param_test1 = {'n_estimators':[10,20,30,40,50,60,70,80,90,100]}
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(
#                 oob_score=True, random_state=10),
#                        param_grid = param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.best_index_, gsearch1.best_params_, gsearch1.best_score_)
#
# param_test2 = {'max_depth':[3,5,7,9,11,13,15,17,19,21]}
# gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
#                                  oob_score=True, random_state=10),
#    param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(X,y)
# print(gsearch2.best_index_, gsearch2.best_params_, gsearch2.best_score_)
#
# param_test4 = {'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}
# gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=5,
#                                  oob_score=True, random_state=10),
#    param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,y)
# print(gsearch4.best_index_, gsearch4.best_params_, gsearch4.best_score_)
#
# param_test3 = {'max_features':[1,2,3,5,7,9,11,13,15,17,19,21]}
# gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=5,min_samples_leaf=10,
#                                  oob_score=True, random_state=10),
#    param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,y)
# print(gsearch3.best_index_, gsearch3.best_params_, gsearch3.best_score_)


# 5 {'n_estimators': 60} 0.6730939716312057
# 1 {'max_depth': 5} 0.6841666666666667
# 9 {'min_samples_leaf': 10} 0.7072222222222221
# 7 {'max_features': 13} 0.741111111111111



# rf2 = RandomForestClassifier(n_estimators= 60,max_depth=5,min_samples_leaf=10,max_features=13,
#                                  oob_score=True, random_state=10)
# rf2.fit(X,y)
# print (rf2.oob_score_)
#
#
y_ = y.as_matrix()
x_ = X.as_matrix()
X_train,X_test,Y_train,Y_test = train_test_split(x_, y_, test_size=0.2,random_state=8)
print ("训练数据条数:%d；数据特征个数:%d；测试数据条数:%d" % (X_train.shape[0], X_train.shape[1], X_test.shape[0]))

pipe = Pipeline([
            ('Poly', PolynomialFeatures(degree = 2)),#
            ('RFC', RandomForestClassifier(
                                  random_state=10))
        ])

pipe.fit(X_train, Y_train)
r = pipe.score(X_train, Y_train)
r_test = pipe.score(X_test, Y_test)
print ("R值：", r)
print ("R_test值：", r_test)
