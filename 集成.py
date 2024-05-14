import random

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载乳腺癌数据集
from sklearn.tree import DecisionTreeClassifier
'''load_breast_cancer() 是一个用于加载乳腺癌数据集的函数，通常是在 Python 的机器学习库 scikit-learn 中使用的。这个数据集包含了乳腺癌诊断的一些特征数据，以及对应的诊断结果（良性或恶性）。这些特征包括肿块的大小、形状、质地等，可以用来训练机器学习模型以预测乳腺肿瘤的良性或恶性。'''
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train)

# 初始化多层SVM作为基分类器
li = [
     SVC(kernel='linear', probability=True, random_state=random.randint(0,100)),
      LogisticRegression(max_iter=1000, random_state=random.randint(0,100)),
      SVC(kernel='rbf', probability=True, random_state=random.randint(0,100)),
      DecisionTreeClassifier(random_state=random.randint(0,100)),
      SVC(kernel='poly', probability=True, random_state=random.randint(0,100)),
      SVC(kernel='sigmoid', probability=True, random_state=random.randint(0,100)),
      LogisticRegression(max_iter=1000, random_state=random.randint(0,100)),
      DecisionTreeClassifier(random_state=random.randint(0,100))
      ]
base_classifiers = []
for i in range(1):
    base_classifiers.append(li[i%len(li)])

# 初始化AdaBoost分类器，将多个基分类器传递给它
adaboost_classifier = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)

# 循环添加多个基分类器
for classifier in base_classifiers:
    adaboost_classifier.set_params(base_estimator=classifier)
    adaboost_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = adaboost_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost分类器在测试集上的准确率：", accuracy)

'''
线性核支持向量机（SVC）：

SVC(kernel='linear', probability=True, random_state=42)：使用线性核的支持向量机分类器，设置了 probability=True 以便在训练后获取类别概率，random_state=42 用于控制随机性，确保结果可重复。
逻辑回归（Logistic Regression）：

LogisticRegression(max_iter=1000, random_state=42)：逻辑回归分类器，设置了 max_iter=1000 来增加迭代次数以确保模型收敛，random_state=42 用于控制随机性。
高斯核支持向量机（SVC）：

SVC(kernel='rbf', probability=True, random_state=42)：使用高斯核的支持向量机分类器，同样设置了 probability=True 和 random_state=42。
决策树（Decision Tree）：

DecisionTreeClassifier(random_state=42)：决策树分类器，设置了 random_state=42 用于控制随机性。
多项式核支持向量机（SVC）：

SVC(kernel='poly', probability=True, random_state=42)：使用多项式核的支持向量机分类器，同样设置了 probability=True 和 random_state=42。
Sigmoid核支持向量机（SVC）：

SVC(kernel='sigmoid', probability=True, random_state=42)：使用Sigmoid核的支持向量机分类器，同样设置了 probability=True 和 random_state=42。
另一个逻辑回归（Logistic Regression）：

LogisticRegression(max_iter=1000, random_state=42)：另一个逻辑回归分类器，与第二个逻辑回归分类器参数设置相同。
另一个决策树（Decision Tree）：

DecisionTreeClassifier(random_state=42)：另一个决策树分类器，与第四个决策树分类器参数设置相同。

AdaBoost分类器在测试集上的准确率： 0.9385964912280702
20次：AdaBoost分类器在测试集上的准确率： 0.9385964912280702
四个不同的svm20次:AdaBoost分类器在测试集上的准确率： 0.631578947368421
高斯svm20次:AdaBoost分类器在测试集上的准确率： 0.9473684210526315
高斯3次:AdaBoost分类器在测试集上的准确率： 0.9473684210526315
线性svm+逻辑+高斯svm:AdaBoost分类器在测试集上的准确率： 0.9473684210526315
1个高斯AdaBoost分类器在测试集上的准确率： 0.9473684210526315
1个线性AdaBoost分类器在测试集上的准确率： 0.8333333333333334
八层AdaBoost分类器在测试集上的准确率： 0.9385964912280702
七层AdaBoost分类器在测试集上的准确率： 0.9736842105263158
六层AdaBoost分类器在测试集上的准确率： 0.631578947368421
五层AdaBoost分类器在测试集上的准确率： 0.6403508771929824
四层AdaBoost分类器在测试集上的准确率： 0.9385964912280702
三层AdaBoost分类器在测试集上的准确率： 0.9473684210526315
两层AdaBoost分类器在测试集上的准确率： 0.9736842105263158
'''
