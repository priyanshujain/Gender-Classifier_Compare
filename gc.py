import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# classifiers


clf_knn =    KNeighborsClassifier(3)
clf_sv1 =    SVC(kernel="linear", C=0.025)
clf_sv2 =    SVC(gamma=2, C=1)
clf_gpc =    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
clf_dtc =    DecisionTreeClassifier(max_depth=5)
clf_rfc =    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf_abc =    AdaBoostClassifier()
clf_gnb =    GaussianNB()
clf_qda =    QuadraticDiscriminantAnalysis()
clf_prc =    Perceptron()


clf_knn = clf_knn.fit(X, Y)
clf_sv1 = clf_sv1.fit(X, Y)
clf_sv2 = clf_sv2.fit(X, Y)
clf_gpc = clf_gpc.fit(X, Y)
clf_dtc = clf_dtc.fit(X, Y)
clf_rfc = clf_rfc.fit(X, Y)
clf_abc = clf_abc.fit(X, Y)
clf_gnb = clf_gnb.fit(X, Y)
clf_qda = clf_qda.fit(X, Y)
clf_prc = clf_prc.fit(X, Y)


clf = [clf_knn,
    clf_sv1,
    clf_sv2,
    clf_gpc,
    clf_dtc,
    clf_rfc,
    clf_abc,
    clf_gnb,
    clf_qda,
    clf_prc]

for it in clf:
    prediction = it.predict(X)
    acc = accuracy_score(Y, prediction)
    print(acc*100)
