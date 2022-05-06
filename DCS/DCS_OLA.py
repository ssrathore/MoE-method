import os
import time
import statistics
from deslib.dcs.ola import OLA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix
#import helpers as hlp   #helpers.py requires
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

dataset_name = [#"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\Ant-1.7.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\Camel-1.2.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\Camel-1.4.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\camel-1.6.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\ivy-2.0.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\jedit-4.3.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\poi-3.0.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\synapse-1.2.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\velocity-1.6.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\xalan-2.4.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\xalan-2.5.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\xalan-2.6.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\xalan-2.7.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\xerces-1.4.csv", "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\CM1.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\eclipse-2.0.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\eclipse-2.1.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\eclipse-3.0.csv", "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\Equinox_Framework.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\JDT_Core.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\JM1.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\KC1.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\KC2.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\KC3.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\Lucene.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\MC1.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\MC2.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\MW1.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\mylyn.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\PC1.csv", "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\PC2.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\PC3.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\PC4.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\PC5.csv", "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\hbase-0.94.0.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\hive-0.9.0.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\jruby-1.1.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\wicket-1.3.0-beta2.csv", "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\prop-1.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\prop-2.csv", "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\prop-3.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\prop-4.csv","D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\prop-5.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_Datasets\\prop-6.csv",
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_OLA_Datasets\\activemq-5.0.0.csv"
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_OLA_Datasets\\PDE_UI.csv"
                #"D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_OLA_Datasets\\groovy-1_6_BETA_1.csv"
                "D:\\Office_PC\\Research_work\\MoE\\Source_Code_local_rerun\\DCS_OLA_Datasets\\derby-10.5.1.1.csv"]


for d in dataset_name:
    result = pd.read_csv(d)
    Number_of_instances,Attributes = result.shape
    X = result.iloc[:, :-1].values
    y = result.iloc[:, -1].values
    index = 0
    for x in range (len(y)):
        if y[x] >= 1:
            y[index] = 1
        index += 1
    Y = np.array(y)
    
    cwd = os.getcwd()
    outfile = open(d +".csv" , "a+")
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    acc = []
    prec = []
    rec = []
    f1 = []
    AUC = []
    TP=0
    TN=0
    FP=0
    FN=0
    for train_index, test_index in kf.split(X):
        X_train1, X_test = X[train_index], X[test_index]
        Y_train1, Y_test = Y[train_index], Y[test_index]
        sm = SMOTE(random_state=12)
        X_train, Y_train = sm.fit_sample(X_train1, Y_train1)
   
# define classifiers to use in the pool
    classifiers = [LogisticRegression(), DecisionTreeClassifier(), GaussianNB(), MLPClassifier()]
# fit each classifier on the training set
    for c in classifiers:
        c.fit(X_train, Y_train)
# define the DCS-LA model
    model = OLA(pool_classifiers=classifiers)
# fit the model
    model.fit(X_train, Y_train)
# make predictions on the test set
    yhat = model.predict(X_test)
# evaluate predictions
    print(confusion_matrix(yhat,Y_test))
    for i in range(len(yhat)): 
        if yhat[i]==Y_test[i]==1:
            TP += 1
        if yhat[i]==1 and Y_test[i]!=yhat[i]:
            FP += 1
        if Y_test[i]==yhat[i]==0:
            TN += 1
        if yhat[i]==0 and Y_test[i]!=yhat[i]:
            FN += 1

        acc.append(accuracy_score(Y_test, yhat))
        prec.append(precision_score(Y_test, yhat, pos_label=1))
        rec.append(recall_score(Y_test, yhat, pos_label=1))
        f1.append(f1_score(Y_test, yhat, pos_label=1))
        AUC.append(roc_auc_score(Y_test, yhat))
    outfile.write(str(round(statistics.mean(acc),4)) +
                  "," + str(round(statistics.mean(f1),4)) +
                  "," + str(round(statistics.mean(prec),4)) +
                  "," + str(round(statistics.mean(rec),4)) +
                  "," + str(round(statistics.mean(AUC),4)) +
                  "," + str(TP) +
                  "," + str(FP) +
                  "," + str(FN) +
                  "," + str(TN) + "\n")
    outfile.close()    
