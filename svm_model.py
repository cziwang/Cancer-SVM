import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

"""
SVM (Single Vector Machine) is a supervised ML model that
uses classification algorithms for two-group classifications.

Geometrically, SVM can be interpreted as a line/hyper-plane between 
two classes that maximizes margin.
"""

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
