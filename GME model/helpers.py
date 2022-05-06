import os
import numpy as np
import random
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import customeMLP as CMLP
from sklearn.model_selection import RandomizedSearchCV

def top10FusingRF(X,Y):
    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(X,Y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[:-10]  #indices top 10 features
    #print(indices)
    #print(indice.shape)
    X = np.delete(X,indices,1)
    return X

def convert_vec2matrix(Y,k):
    y = np.zeros([Y.shape[0] , k])
    for idx in range(Y.shape[0]):
        y[idx,Y[idx]] = 1
    return y

def reduced_features(data):
    #var_threshold = 0.95
    #pca = decomposition.PCA()
    pca = decomposition.KernelPCA(kernel = "rbf")
    #Reduce features by 5
    #pca.n_components = data.shape[1] - 5
    pca_data = pca.fit_transform(data)
    """percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)
    reduced_features = 0
    for value in cum_var_explained:
        reduced_features += 1
        if value >= var_threshold:
            break
    #print(pca_data)
    reduced_data = pca_data[: ,:reduced_features]"""
    return pca_data

def checkvariance(X):
    #for i in range(X.shape[1]):
        #print(X[: ,i:i+1].var())
    print(np.corrcoef(X))

def loaddata(dataset_name,
             sampling_required="yes",
             standardisation="yes",
             pca_reduced="yes",
             normalization="yes"):

    cwd = os.getcwd()
    data = np.genfromtxt(cwd + "/Datasets/" + dataset_name +".csv", delimiter=',')
    #data = np.genfromtxt(cwd + dataset_name +".csv", delimiter=',')
    print(data)
    #data[np.isnan(data)] = 0
    data = data[~np.isnan(data).any(axis=1)]   #Remove rows having NaN
    
    Number_of_instances,Attributes = data.shape
    X = np.array([x[:-1] for x in data])
    y = [0 for i in range(Number_of_instances)]
    
    index = 0
    for x in data:
        if int(x[-1:]) >= 1:
            y[index] = 1
        index += 1
    Y = np.array(y)

    #X = top10FusingRF(X,Y)
    
    if sampling_required == "yes":
        X, Y = SMOTE().fit_resample(X, Y)

    if standardisation == "yes":
        X = StandardScaler().fit_transform(X)

    if normalization == "yes":
        X = normalize(X)
        
    if pca_reduced == "yes":
        X = reduced_features(X)
    
    c = ['Green' for i in range(Y.shape[0])]
    index = 0
    for x in Y:
        if x == 1:
            c[index] = 'Red'
        index += 1
    color = np.array(c)

    return X,Y,color

#GM model based ME
def ME_model( X_train, Y_train, X_test, n_estimator, expert_model, _DS, _LP):

    models = []
    if expert_model == "DT":
        
        models = [DecisionTreeClassifier() for i in range(n_estimator)]
    elif expert_model == "MLP":
        
        models = [MLPClassifier(max_iter=200) for i in range(n_estimator)]
        #models = [mlp for i in range(n_estimator)]
    elif expert_model == "NB":
        models = [GaussianNB() for i in range(n_estimator)]
    elif expert_model == "LR":
        models = [LogisticRegression() for i in range(n_estimator)]
    
    gm = GaussianMixture(n_components = n_estimator, init_params = 'random',reg_covar=1e-05)
    gm.fit(X_train)
    train_prob = gm.predict_proba(X_train)

    x = [ None ] * n_estimator
    y = [ None ] * n_estimator
    del_e = [ None ] * n_estimator

    for j in range(n_estimator):
        del_e[j] = np.ndarray.tolist(np.arange(X_train.shape[0]))
        x[j] = X_train
        y[j] = Y_train


    for i in range(X_train.shape[0]):
        for j in range(n_estimator):
            if train_prob[i][j] > _DS:
                del_e[j].remove(i)

    for j in range(n_estimator):
        x[j] = np.delete(x[j],del_e[j],0)
        y[j] = np.delete(y[j],del_e[j],0)
    #Done soft split of train data

    #Training Model
    for j in range(n_estimator):
        models[j].fit(x[j],y[j])

    #Testing model
    test_prob = gm.predict_proba(X_test)
    Y_pred = np.array([0 for _ in range(X_test.shape[0])])
    for i in range(X_test.shape[0]):
        temp = 0
        for j in range(n_estimator):
            y = models[j].predict(X_test[i:i+1, :])
            temp += test_prob[i][j] * y
        if temp > _LP:
            Y_pred[i] = 1
    return Y_pred


#Bagging ensemble
def bag_model(X_train, Y_train, X_test, expert_model, n_estimator):

    if expert_model == "DT":
        model = BaggingClassifier( DecisionTreeClassifier(),
                                   n_estimators = n_estimator,
                                   max_samples = 0.7,
                                   max_features = 0.7,
                                   bootstrap = True,
                                   bootstrap_features = True)
    if expert_model == "MLP":
        model = BaggingClassifier( MLPClassifier(),
                                   n_estimators = n_estimator,
                                   max_samples = 0.7,
                                   max_features = 0.7,
                                   bootstrap = True,
                                   bootstrap_features = True)
    if expert_model == "NB":
        model = BaggingClassifier( GaussianNB(),
                                   n_estimators = n_estimator,
                                   max_samples = 0.7,
                                   max_features = 0.7,
                                   bootstrap = True,
                                   bootstrap_features = True)
    if expert_model == "LR":
        model = BaggingClassifier( LogisticRegression(),
                                   n_estimators = n_estimator,
                                   max_samples = 0.7,
                                   max_features = 0.7,
                                   bootstrap = True,
                                   bootstrap_features = True)

    model.fit(X_train ,Y_train)

    Y_pred = model.predict(X_test)

    return Y_pred

#Boosting ensemble
def boost_model(X_train, Y_train, X_test, expert_model,n_estimator):

    model = []
    if expert_model == "DT":
        model = AdaBoostClassifier( DecisionTreeClassifier(),
                                   n_estimators = n_estimator,
                                   algorithm='SAMME.R')
    if expert_model == "MLP":
        model = AdaBoostClassifier( CMLP.customMLPClassifer(),
                                   n_estimators = n_estimator,
                                   algorithm='SAMME')
    if expert_model == "NB":
        model = AdaBoostClassifier( GaussianNB(),
                                   n_estimators = n_estimator,
                                   algorithm='SAMME')
    if expert_model == "LR":
        model = AdaBoostClassifier( LogisticRegression(),
                                   n_estimators = n_estimator,
                                   algorithm='SAMME')

    model.fit(X_train ,Y_train)

    Y_pred = model.predict(X_test)

    return Y_pred
    
#Stacking Ensemble
def stack_model(X_train, Y_train, X_test, expert_model,n_estimator):

    model =[]
    estimators = [('DT', DecisionTreeClassifier()),
                 ('MLP', MLPClassifier())]
    if expert_model == "DT":
        model = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())
    if expert_model == "MLP":
        model = StackingClassifier(estimators=estimators, final_estimator=MLPClassifier())
    if expert_model == "NB":
        model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
    if expert_model == "LogisticRegression()":
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    model.fit(X_train ,Y_train)

    Y_pred = model.predict(X_test)

    return Y_pred    


#Single/Individual Model
def indi_model(X_train, Y_train, X_test, expert_model):
    print()
    print("Individual Model: "+expert_model)
    print()
    if expert_model == "DT":
        model = DecisionTreeClassifier()
    if expert_model == "MLP":
        model = MLPClassifier()
    if expert_model == "NB":
        model = GaussianNB()
    if expert_model == "LR":
        model = LogisticRegression()

    model.fit(X_train ,Y_train)

    Y_pred = model.predict(X_test)

    return Y_pred
