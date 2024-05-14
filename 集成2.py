from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载乳腺癌数据集
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化SVM作为基分类器
base_classifier = SVC(kernel='rbf', probability=True, random_state=42)

# 初始化AdaBoost分类器
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, learning_rate=1.0, random_state=42)

# 训练AdaBoost分类器
adaboost_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = adaboost_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost分类器在测试集上的准确率：", accuracy)
