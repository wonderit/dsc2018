import csv

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PowerTransformer

# machine learning

data_train = pd.read_csv("../data/sd_pr_id_mean_likeo_train.csv")
data_test = pd.read_csv("../data/sd_pr_id_mean_likeo_test.csv")
dataset_train = data_train.values
dataset_test = data_test.values

#
# gender,age,age_o,samerace,importance_same_race,pref_o_shared_interests,pref_o_attractive,pref_o_sincere,pref_o_intelligence,pref_o_funny,pref_o_ambitious,attractive_o,sinsere_o,intelligence_o,funny_o,ambitous_o,shared_interests_o,attractive_important,sincere_important,intellicence_important,funny_important,ambtition_important,shared_interests_important,attractive,sincere,intelligence,funny,ambition,attractive_partner,sincere_partner,intelligence_partner,funny_partner,ambition_partner,shared_interests_partner,met,like,att_m,sin_m,int_m,fun_m,amb_m,sha_m
variables_x = [
    'gender',
    'age',
    'age_o',
    'samerace','importance_same_race',
    'pref_o_shared_interests','pref_o_attractive','pref_o_sincere',
    'pref_o_intelligence','pref_o_funny','pref_o_ambitious',
    'attractive_o','sinsere_o','intelligence_o','funny_o','ambitous_o','shared_interests_o',
    'attractive_important','sincere_important','intellicence_important',
    'funny_important','ambtition_important','shared_interests_important',
    'attractive','sincere','intelligence','funny','ambition',
    'attractive_partner','sincere_partner','intelligence_partner',
    'funny_partner','ambition_partner','shared_interests_partner',
    'met',
    'like',
    'att_m','sin_m','int_m','fun_m','amb_m','sha_m'
]

variables_y = [
    'like_o'
]

train_x = dataset_train[:, data_train.columns.isin(variables_x)]
train_y = dataset_train[:, data_train.columns.isin(variables_y)].reshape(-1)
test_x = dataset_test[:, data_train.columns.isin(variables_x)]
test_y = dataset_test[:, data_train.columns.isin(variables_y)].reshape(-1)

#
# train_x = StandardScaler().fit_transform(train_x)
# test_x = StandardScaler().fit_transform(test_x)

# train_x = MinMaxScaler().fit_transform(train_x)
# test_x = MinMaxScaler().fit_transform(test_x)
# train_x = QuantileTransformer().fit_transform(train_x)
# test_x = QuantileTransformer().fit_transform(test_x)
#
train_x = PowerTransformer().fit_transform(train_x)
test_x = PowerTransformer().fit_transform(test_x)

#
myFile = open('../data/power_m_o_test_x.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(test_x)

myFile2 = open('../data/power_m_o_train_x.csv', 'w')
with myFile2:
    writer2 = csv.writer(myFile2)
    writer2.writerows(train_x)


# train_x = PowerTransformer().fit_transform(train_x)
# test_x = PowerTransformer().fit_transform(test_x)

SCALER = None

def run_kfold(clf):
    kf = KFold(n_splits=10)
    outcomes = []
    fold = 0
    xval_err = 0
    for train_index, test_index in kf.split(train_x):
        fold += 1
        Xtrain, Xtest = train_x[train_index], train_x[test_index]
        ytrain, ytest = train_y[train_index], train_y[test_index]
        clf.fit(Xtrain, ytrain)
        predictions = clf.predict(Xtest)
        # accuracy = accuracy_score(ytest, predictions)
        # outcomes.append(accuracy)
        e = predictions - ytest
        xval_err += np.dot(e, e)

    rmse_10cv = np.sqrt(xval_err / len(train_x))
    print('RMSE on 10-fold CV: %.2f' % rmse_10cv)

    prediction_test = clf.predict(test_x)
    e_test = prediction_test - test_y
    xtest_err = 0
    xtest_err += np.dot(e_test, e_test)
    rmse_test = np.sqrt(xtest_err / len(test_x))
    print('TEST RMSE : %.2f' % rmse_test)


names = [
    "Linear Regression",
    "LAsso",
    "Ridge",
    "SGDRegressor",
    "RandomForestRegressor",
    "MLPRegressor"
]

regressors = [
    LinearRegression(),
    Lasso(),
    Ridge(),
    SGDRegressor(),
    RandomForestRegressor(),
    MLPRegressor(activation='identity',
                 hidden_layer_sizes=(5, 2),
                 random_state=1,
                 max_iter=1000
                 )
]


index = 0
# iterate over classifiers
for name, clf in zip(names, regressors):
    index += 1
    print(index, "th model : ", name)
    run_kfold(clf)



# Create linear regression object# Create
# linreg = LinearRegression()
#
# # Train the model using the training sets
# linreg.fit(train_x,train_y)
#
#
# # Compute RMSE on training data# Comput
# # p = np.array([linreg.predict(xi) for xi in x])
# p = linreg.predict(train_x)
# # Now we can constuct a vector of errors
# err = abs(p-train_y)
#
# # Dot product of error vector with itself gives us the sum of squared errors# Dot pr
# total_error = np.dot(err,err)
# # Compute RMSE
# rmse_train = np.sqrt(total_error/len(p))
# print (rmse_train)
#
# # We can view the regression coefficients
# print ('Regression Coefficients: \n', linreg.coef_)
#
# run_kfold(linreg)

