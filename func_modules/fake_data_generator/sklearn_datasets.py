from sklearn.datasets import make_blobs, make_moons, make_circles, make_regression, make_classification
import matplotlib.pyplot as plt
from pandas import  DataFrame
import numpy as np



# 成簇斑点数据（高斯分布）
# 适用于生成线性可分/近似线性可分的多分类数据集
# X, y = make_blobs(n_samples=100, centers=3, n_features=2, center_box=(-10,10), cluster_std=1)
# df =DataFrame(dict(x= X[:,0], y= X[:,1], label=y))
# colors = {0:'red', 1:'blue', 2:'green'}
#
# fig, ax = plt.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     print(key)
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#
# plt.show()


# 卫星分类
# 适用于非线性二分类问题
# X, y = make_moons(n_samples=100, noise=0.1)
# df =DataFrame(dict(x= X[:,0], y= X[:,1], label=y))
# colors = {0:'red', 1:'blue'}
#
# fig, ax = plt.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     print(key)
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#
# plt.show()


# 圆圈分类问题
# 适用于负载的非线性二分类问题
# X, y = make_circles(n_samples=100, noise=0.05)
# df =DataFrame(dict(x= X[:,0], y= X[:,1], label=y))
# colors = {0:'red', 1:'blue'}
#
# fig, ax = plt.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     print(key)
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#
# plt.show()


# 回归数据
# 适用于线性回归问题
# X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
# plt.scatter(X, y)
# plt.show()






