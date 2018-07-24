import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

data_train = pd.read_csv("../data/speeddating_train_preprocessed.csv")
data_test = pd.read_csv("../data/speeddating_test_preprocessed.csv")
dataset_train = data_train.values
dataset_test = data_test.values

train_x = dataset_train[:, :-1]
train_y = dataset_train[:, -1]

test_x = dataset_test[:, :-1]
test_y = dataset_test[:, -1]

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)


def run_kfold(clf):
    kf = KFold(5028, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        Xtrain, Xtest = train_x[train_index], train_x[test_index]
        ytrain, ytest = train_y[train_index], train_y[test_index]
        clf.fit(Xtrain, ytrain)
        predictions = clf.predict(Xtest)
        accuracy = accuracy_score(ytest, predictions)
        outcomes.append(accuracy)
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))


print("### 1. model : RandomForestClassifier")
random_forest = RandomForestClassifier()
run_kfold(random_forest)

print("### 2. model : LogisticRegression")
logreg = LogisticRegression()
run_kfold(logreg)

print("### 3. model : SVC")
svc = SVC()
run_kfold(svc)

print("### 4. model : KNeighborsClassifier")
knn = KNeighborsClassifier(n_neighbors = 8)
run_kfold(knn)

print("### 5. model : DecisionTree")
decisionTree = DecisionTreeClassifier()
run_kfold(decisionTree)

