import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import copy
from sklearn.datasets import make_circles


data = pd.read_csv('breast-cancer-wisconsin_cleaned.csv')

Y = data[['Class']]
X = data.drop('Class', axis=1)
#
# pca = PCA(n_components=2)
# X_new = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)
#
# plt.scatter(X_new[:, 0], X_new[:, 1], c=Y['Class'])
# plt.show()


from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=2, random_state=42)
y_predict = estimator.fit_predict(X)
print(y_predict)
print(sum(np.array(y_predict)==0))
print(sum(np.array(y_predict)==1))
print(Y['Class'].value_counts())

Y_temp = copy.deepcopy(Y)
Y_temp.loc[Y['Class']==2, 'Class'] = 0
Y_temp.loc[Y['Class']==4, 'Class'] = 1
print(accuracy_score(Y_temp['Class'], y_predict))


from sklearn.datasets import make_circles

