import numpy as np
import pandas as pd
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler, PowerTransformer

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score

data_train = pd.read_csv("../data/speeddating_train_preprocessed.csv")
data_test = pd.read_csv("../data/speeddating_test_preprocessed.csv")
dataset_train = data_train.values
dataset_test = data_test.values

train_x = dataset_train[:, :-1]
train_y = dataset_train[:, -1]

test_x = dataset_test[:, :-1]
test_y = dataset_test[:, -1]
#
train_x = StandardScaler().fit_transform(train_x)
test_x = StandardScaler().fit_transform(test_x)
#
# train_x = MinMaxScaler().fit_transform(train_x)
# test_x = MinMaxScaler().fit_transform(test_x)

# train_x = QuantileTransformer().fit_transform(train_x)
# test_x = QuantileTransformer().fit_transform(test_x)

# train_x = PowerTransformer().fit_transform(train_x)
# test_x = PowerTransformer().fit_transform(test_x)


def run_kfold(clf):
    kf = KFold(n_splits=10, shuffle=True)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(train_x):
        fold += 1
        Xtrain, Xtest = train_x[train_index], train_x[test_index]
        ytrain, ytest = train_y[train_index], train_y[test_index]
        clf.fit(Xtrain, ytrain)
        predictions = clf.predict(Xtest)
        accuracy = accuracy_score(ytest, predictions)
        outcomes.append(accuracy)
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))


names = ["Nearest Neighbors",
         "Logistic Regression",
         "Linear SVM",
         "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA"]

classifiers = [
    KNeighborsClassifier(n_neighbors=8),
    LogisticRegression(),
    LinearSVC(),
    SVC(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(activation='relu',
                  solver='lbfgs',
                  alpha=1e-5,
                  hidden_layer_sizes=(5, 2),
                  random_state=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


index = 0
# iterate over classifiers
for name, clf in zip(names, classifiers):
    index += 1
    print(index, "th model : ", name)
    run_kfold(clf)



