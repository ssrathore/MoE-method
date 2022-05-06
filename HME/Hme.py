import numpy as np
import matplotlib.pyplot as plt
import general_hme as hm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import os
import time
import statistics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle
    

dataset_name = [#"D:\MoE\Source_Code\Datasets\Ant-1.7.csv"
                #"D:\MoE\Source_Code\Datasets\Camel-1.2.csv",
                #"D:\MoE\Source_Code\Datasets\Camel-1.4.csv","D:\MoE\Source_Code\Datasets\Camel-1.6.csv",
                #"D:\MoE\Source_Code\Datasets\CM1.csv",
                #"D:\MoE\Source_Code\Datasets\eclipse-2.0.csv",
                #"D:\MoE\Source_Code\Datasets\eclipse-2.1.csv",
                #"D:\MoE\Source_Code\Datasets\eclipse-3.0.csv",
                #"D:\MoE\Source_Code\Datasets\Equinox_Framework.csv",
                #"D:\MoE\Source_Code\Datasets\JM1.csv",
                #"D:\MoE\Source_Code\Datasets\JDT_Core.csv",
                #"D:\MoE\Source_Code\Datasets\KC1.csv","D:\MoE\Source_Code\Datasets\KC2.csv",
                #"D:\MoE\Source_Code\Datasets\KC3.csv",
                #"D:\MoE\Source_Code\Datasets\Lucene.csv",
                #"D:\MoE\Source_Code\Datasets\MC1.csv",
                #"D:\MoE\Source_Code\Datasets\MC2.csv",
                #"D:\MoE\Source_Code\Datasets\MW1.csv",
                #"D:\MoE\Source_Code\Datasets\ivy-2.0.csv","D:\MoE\Source_Code\Datasets\jedit-4.3.csv",
                #"D:\MoE\Source_Code\Datasets\mylyn.csv",
                #"D:\MoE\Source_Code\Datasets\PC1.csv",
                #"D:\MoE\Source_Code\Datasets\PC2.csv",
                #"D:\MoE\Source_Code\Datasets\PC31.csv",
                #"D:\MoE\Source_Code\Datasets\PC4.csv",
                #"D:\MoE\Source_Code\Datasets\PC5.csv",
                "D:\MoE\Source_Code\Datasets\PDE_UI.csv",
                #"D:\MoE\Source_Code\Datasets\poi-3.0.csv",
                #"D:\MoE\Source_Code\Datasets\prop-1.csv",
                #"D:\MoE\Source_Code\Datasets\prop-2.csv",
                #"D:\MoE\Source_Code\Datasets\prop-3.csv","D:\MoE\Source_Code\Datasets\prop-4.csv","D:\MoE\Source_Code\Datasets\prop-5.csv","D:\MoE\Source_Code\Datasets\prop-6.csv",
                #"D:\MoE\Source_Code\Datasets\synapse-1.2.csv",
                #"D:\MoE\Source_Code\Datasets\Velocity-1.6.csv",
                #"D:\MoE\Source_Code\Datasets\Axalan-2.4.csv",
                #"D:\MoE\Source_Code\Datasets\Axalan-2.5.csv","D:\MoE\Source_Code\Datasets\Axalan-2.6.csv","D:\MoE\Source_Code\Datasets\Axalan-2.7.csv",
                #"D:\MoE\Source_Code\Datasets\Axerces-1.4.csv"
                ]
for d in dataset_name:
    result = pd.read_csv(d,header=None)
    cols = pd.read_csv(d, header=None, nrows=1).columns
    print(cols)
    df = pd.read_csv(d, header=None, usecols=cols[:-1])
    #X = result.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
                       #51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
                       #101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120, 121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,
                       #139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154]].values
    #X = result.iloc[0:-1]].values
    X=df.values
    Y= result.iloc[:,17].values
    
    Number_of_instances,Attributes = result.shape
    index = 0
    for i in range(Number_of_instances):
        if Y[index] > 1:
            Y[index]= 1
        index +=1
        
    X, Y = SMOTE().fit_resample(X, Y)
    X, Y = shuffle(X, Y)
    X = StandardScaler().fit_transform(X)
    print(X)
    print(Y)
    print(index)
    print(Number_of_instances)
    cwd = os.getcwd()
    outfile = open(d +".csv" , "a+")

    kf = KFold(n_splits=5, random_state=None, shuffle=True) #5-fold CV
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
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

   # fit hme model
        hme   = hm.HME(Y_train,X_train,Y_test,X_test,"softmax",bias = True, gate_type = "softmax", verbose = False, levels = 6, branching = 2)
        hme.fit()
        hme_predict = hme.predict(X_test)

        print("\n HME \n")
        print(confusion_matrix(hme_predict,Y_test))

        for i in range(len(hme_predict)): 
            if hme_predict[i]==Y_test[i]==1:
                TP += 1
            if hme_predict[i]==1 and Y_test[i]!=hme_predict[i]:
                FP += 1
            if Y_test[i]==hme_predict[i]==0:
               TN += 1
            if hme_predict[i]==0 and Y_test[i]!=hme_predict[i]:
                FN += 1

        acc.append(accuracy_score(Y_test, hme_predict))
        prec.append(precision_score(Y_test, hme_predict, pos_label=1))
        rec.append(recall_score(Y_test, hme_predict, pos_label=1))
        f1.append(f1_score(Y_test, hme_predict, pos_label=1))
        AUC.append(roc_auc_score(Y_test, hme_predict))
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
    
    
'''# grid
x1         = np.linspace(0,1,500)
x2         = np.linspace(0,1,500)
X1,X2      = np.meshgrid(x1,x2)
Xgrid      = np.zeros([250000,2])
Xgrid[:,0] = np.reshape(X1,(250000,))
Xgrid[:,1] = np.reshape(X2,(250000,))
    
# predict hme probs on grid
hme_grid_predict = hme.predict(Xgrid, predict_type = "predict_probs")
    
# plot grid
plt.figure(figsize = (7,7))
plt.contourf(X1,X2,np.reshape(hme_grid_predict[:,0],(500,500)), cmap="coolwarm")
plt.plot(X_test[Y_test=="y",0],X_test[Y_test=="y",1],"bo", markersize = 3)
plt.plot(X_test[Y_test=="n",0],X_test[Y_test=="n",1],"ro", markersize = 3)
plt.colorbar()
title = "HME decision boundaries on test set"
plt.title("Probabilities HME")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
'''
