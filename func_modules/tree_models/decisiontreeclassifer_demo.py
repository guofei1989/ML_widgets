from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


data = load_iris()
train_X, test_X, train_y, test_y = train_test_split(data["data"][:, [0, 2]], data["target"], test_size=0.3)

# 训练模型
clf = RandomForestClassifier(max_depth=4, oob_score=True)
# clf = DecisionTreeClassifier(max_depth=4)
clf.fit(train_X, train_y)

# 画图
x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)     # 预测值
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=0.8)     # 样本值
plt.show()

print(clf.oob_score_)

# 预测模型
pred_y = clf.predict(test_X)
print(confusion_matrix(test_y, pred_y))




