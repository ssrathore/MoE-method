import os
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
import sys

def nOInst(data):
    return data.shape[0]

def nOAttr(data):
    return data.shape[1]

def nOFInst(data):
    count = 0
    for x in data:
        if x == 1:
            count += 1

    return count

def pOFInst(data):
    t_pts = nOInst(data)
    f_pts = nOFInst(data)
    pfi = (f_pts / t_pts) *100
    return round(pfi,2)

def nONaN(data):
    count = 0
    for i in data:
        for j in i:
            if np.isnan(j) :
                count+=1
    return count

dataset_name = sys.argv[1]
#dataset_name = "JM1"

cwd = os.getcwd()
outfile = open(cwd + "/Results/New/dataset_description.csv" , "a+")

c1 = dataset_name

np_data = np.genfromtxt(cwd + "/Dataset/" + dataset_name +".csv", delimiter=',')
c4 = nONaN(np_data)


np_data = np_data[~np.isnan(np_data).any(axis=1)]
c2 = nOInst(np_data)
X = np.array([x[:-1] for x in np_data])
y = [0 for i in range(c2)]
index = 0
for x in np_data:
    if int(x[-1:]) == 1:
        y[index] = 1
    index += 1
Y = np.array(y)

c3 = nOAttr(X)
c5 = nOFInst(Y)
c6 = pOFInst(Y)

X, Y = SMOTE().fit_resample(X, Y)
c7 = nOInst(X)

outfile.write(c1+','+str(c2)+','+str(c3)+','+str(c4)+','
              +str(c5)+','+str(c6)+','+str(c7)+'\n')


outfile.close()

