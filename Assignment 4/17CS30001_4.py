#ROLL NO. : 17CS30001
#NAME     : ABHIK NASKAR
#Assignment Number : 04
#Compile by running python3.17CS30001_4.py on terminal


import numpy as np
import pandas as pd
import random as rd

df = pd.read_csv('data4_19.csv')

df.Type[df.Type == 'Iris-setosa'] = 1
df.Type[df.Type == 'Iris-versicolor'] = 2
df.Type[df.Type == 'Iris-virginica'] = 3

iris_count = [0, 0, 0]

type_count = [0 ,0 ,0]


X_features = df.iloc[:, [0,1, 2, 3]].values

for i in range(X_features.shape[0]):
    if(df.Type[i] == 1):
        iris_count[0] = iris_count[0] + 1
    elif(df.Type[i] == 2):
        iris_count[1] = iris_count[1] + 1
    else:
        iris_count[2] += 1




m = X_features.shape[0]
n = X_features.shape[1]

n_iter = 10

K = 3

Centroids = np.array([]).reshape(n, 0)
for i in range(K):
    rand = rd.randint(0, m-1)
    Centroids = np.c_[Centroids, X_features[rand]]

#print("Centroids :" + str(Centroids.T))

Output = {}
EuclidianDistance = np.array([]).reshape(m, 0)
for k in range(K):
    tempDist = np.sum((X_features - Centroids[:, k])**2, axis = 1)
    EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    
C = np.argmin(EuclidianDistance, axis= 1) + 1

Y = {}
for k in range(K):
    Y[k+1] = np.array([]).reshape(n, 0)

for i in range(m):
    Y[C[i]] = np.c_[Y[C[i]], X_features[i]]
    
for k in range(K):
    Y[k+1] = Y[k+1].T
    
for k in range(K):
    Centroids[:, k] = np.mean(Y[k+1], axis=0)


for i in range(n_iter):
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
        tempDist=np.sum((X_features-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        
    C=np.argmin(EuclidianDistance,axis=1)+1
    
    Y={}
    index = [0]*m
    for k in range(K):
        Y[k+1]=np.array([]).reshape(n,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X_features[i]]
        index[i] = C[i]
    for k in range(K):
        Y[k+1]=Y[k+1].T
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
        
    Output = Y

type_count[0] = len(Output[1])
type_count[1] = len(Output[2])
type_count[2] = len(Output[3])

print("Final Mean for Cluster 1 : " + str(Centroids[:,0]))
print("Final Mean for Cluster 2 : " + str(Centroids[:,1]))
print("Final Mean for Cluster 3 : " + str(Centroids[:,2]))

#print(Output)

jac = [0, 0, 0]
iris = ['', '', '']





iris_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#print(iris_name)

for i in range(K):
    max_jac = -1
    for j in range(K):
        inter = 0
        for m in range(X_features.shape[0]):
            if((df.Type[m] == j + 1) and (index[m] == i + 1)):
                inter += 1

        union = type_count[i] + iris_count[j] - inter
        jac_index = inter/union
        if(max_jac < jac_index):
            max_jac = jac_index
            iris[i] = iris_name[j]

    jac[i] = 1 - max_jac


for i in range(len(iris)):
    print("Jaccard distance for " + str(i+1) + " cluster is: " + str(jac[i]))




