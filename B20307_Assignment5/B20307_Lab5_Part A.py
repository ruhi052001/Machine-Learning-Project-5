# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statistics as st
# filter warning
import warnings

warnings.filterwarnings("ignore")

# Lab 5 Part A
# Question 1 and Question 2
# import file
df = pd.read_csv('SteelPlateFaults-2class.csv')
# column list
col_list = list(df.columns)
# split list in train and test
[df1_train, df1_test] = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

# copy of test and train data
traingmm = df1_train.copy(deep=True)
testgmm = df1_test.copy(deep=True)
# dataframe of test Class
testclass = testgmm['Class']
# drop bad columns
traingmm.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400', 'X_Minimum', 'Y_Minimum'], inplace=True)
testgmm.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400', 'X_Minimum', 'Y_Minimum', 'Class'], inplace=True)
# divide train data
traingmm0 = traingmm[traingmm["Class"] == 0]
traingmm1 = traingmm[traingmm["Class"] == 1]
# drop Class column in both divided dataframes
traingmm0.drop(columns=['Class'], inplace=True)
traingmm1.drop(columns=['Class'], inplace=True)
# list of accuracies
accs = []
for i in (2, 4, 8, 16):
    # fit train data with class 0 in GMM0 with i clusters
    GMM0 = GaussianMixture(n_components=i, covariance_type='full')
    GMM0.fit(traingmm0.values)
    # fit train data with class 1 in GMM1 with i clusters
    GMM1 = GaussianMixture(n_components=i, covariance_type='full')
    GMM1.fit(traingmm1.values)
    # find score for test data with both class 0 and 1
    t0 = GMM0.score_samples(testgmm.values)
    t1 = GMM1.score_samples(testgmm.values)
    # list of predicted class
    p = []
    # append class in list based on score sample of class 0 and 1
    for j in range(len(t0)):
        if t0[j] >= t1[j]:
            p.append(0)
        elif t0[j] <= t1[j]:
            p.append(1)
            # find accuracy and confusion matrix for each case
    mat = confusion_matrix(testclass, p)
    acc = accuracy_score(testclass, p)
    # append accuracy in list
    accs.append(acc)
    # print result
    print("Confusions matrix with Q=", i, ":")
    print(mat)
    print("Accuracy with Q=", i, ":", acc)

# print max accuracy for each model
print("----------Q2-------------")
print("Max accuracy for KNN", 0.896)
print("Max accuracy for KNN Normalized", 1.0)
print("Accuracy of Bayes Classifier", 0.9375)
print("Max Accuracy in GMM", round(max(accs), 3))
