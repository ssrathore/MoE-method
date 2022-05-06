import os
import time
import statistics
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
from scipy.stats import wilcoxon
from statistics import mean, stdev
from math import sqrt


x = pd.read_csv("D:\Office_PC\Research_work\MoE\Source_Code_local_rerun\Results_Sum\Wilcoxon.csv", index_col=None, usecols = ['Bagging-DT'])
y = pd.read_csv("D:\Office_PC\Research_work\MoE\Source_Code_local_rerun\Results_Sum\Wilcoxon.csv", index_col=None, usecols = ['GME-DT'])
X=x.values.flatten()
Y=y.values.flatten()
#Y=np.array(y)
#print(X.shape)
#print(X)
#print(Y)
d, p = wilcoxon(X, Y, zero_method='wilcox', correction=True, alternative='less', mode='auto')
cohens_d = abs((mean(X) - mean(Y)) / (sqrt((stdev(X) ** 2 + stdev(Y) ** 2) / 2)))
print(cohens_d)
print(p)
