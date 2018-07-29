import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

data_train = pd.read_csv("../data/speeddating_train_preprocessed.csv")
data_test = pd.read_csv("../data/speeddating_test_preprocessed.csv")
dataset_train = data_train.values
dataset_test = data_test.values


variables_x = [
    'gender',
    'age',
    'age_o',
    'samerace','importance_same_race',
    'pref_o_shared_interests','pref_o_attractive','pref_o_sincere',
    'pref_o_intelligence','pref_o_funny','pref_o_ambitious',
    # 'attractive_o','sinsere_o','intelligence_o','funny_o','ambitous_o','shared_interests_o',
    'attractive_important','sincere_important','intellicence_important',
    'funny_important','ambtition_important','shared_interests_important',
    'attractive','sincere','intelligence','funny','ambition',
    'attractive_partner','sincere_partner','intelligence_partner',
    'funny_partner','ambition_partner','shared_interests_partner',
    'met',
    'like',
    # 'att_m','sin_m','int_m','fun_m','amb_m','sha_m'
]

variables_y = [
    'decision_o'
]

train_x = dataset_train[:, data_train.columns.isin(variables_x)]
train_y = dataset_train[:, data_train.columns.isin(variables_y)].reshape(-1)
test_x = dataset_test[:, data_train.columns.isin(variables_x)]
test_y = dataset_test[:, data_train.columns.isin(variables_y)].reshape(-1)
#

SCALER = MinMaxScaler()

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
    print("K fold model Mean Accuracy: %.1f" % (mean_outcome * 100))
    predictions_test = clf.predict(test_x)
    accuracy_test = accuracy_score(test_y, predictions_test)
    print("K fold model TEST Accuracy: %.1f" % (accuracy_test * 100))


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
    Pipeline([('scaler', SCALER), ('classifier', KNeighborsClassifier(n_neighbors=8))]),
    Pipeline([('scaler', SCALER), ('classifier', LogisticRegression())]),
    Pipeline([('scaler', SCALER), ('classifier', LinearSVC(max_iter=5000, random_state=1))]),
    Pipeline([('scaler', SCALER), ('classifier', SVC(gamma='scale'))]),
    Pipeline([('scaler', SCALER), ('classifier', DecisionTreeClassifier(max_depth=5))]),
    Pipeline([('scaler', SCALER), ('classifier', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))]),
    Pipeline([('scaler', SCALER), ('classifier', MLPClassifier(activation='relu',
                      solver='lbfgs',
                      alpha=1e-5,
                      hidden_layer_sizes=(5, 2),
                      random_state=1))]),
    Pipeline([('scaler', SCALER), ('classifier', AdaBoostClassifier())]),
    Pipeline([('scaler', SCALER), ('classifier', GaussianNB())]),
    Pipeline([('scaler', SCALER), ('classifier', QuadraticDiscriminantAnalysis())]),
]


index = 0
# iterate over classifiers
for name, clf in zip(names, classifiers):
    index += 1
    print(index, "th model : ", name)
    run_kfold(clf)



