#ROLL NO. : 17CS30001
#NAME     : ABHIK NASKAR
#Assignment Number : 04
#Compile by running python3.17CS30001_4.py on terminal


import numpy as np
import pandas as pd
import random as rd

df = pd.read_csv('/Users/abhiknaskar/Desktop/Machine Learning IIT KGP/data4_19.csv')

df.Type[df.Type == 'Iris-setosa'] = 1
df.Type[df.Type == 'Iris-versicolor'] = 2
df.Type[df.Type == 'Iris-virginica'] = 3

X_features = df.iloc[:, [0,1, 2, 3]].values

m = X_features.shape[0]
n = X_features.shape[1]

n_iter = 10

K = 3

Centroids = np.array([]).reshape(n, 0)
for i in range(K):
    rand = rd.randint(0, m-1)
    Centroids = np.c_[Centroids, X_features[rand]]

print("Centroids :" + str(Centroids.T))

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
    for k in range(K):
        Y[k+1]=np.array([]).reshape(n,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X_features[i]]
    for k in range(K):
        Y[k+1]=Y[k+1].T
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
        
    Output = Y

print("Final Mean for Cluster 1 : " + str(Centroids[:,0]))
print("Final Mean for Cluster 2 : " + str(Centroids[:,1]))
print("Final Mean for Cluster 3 : " + str(Centroids[:,2]))

#print(Output)

for k in range(K):
    count = 0
    for i in Output[k+1]:
        #print(i)
        for j in range(m):
            if((i == X_features[j]).all()):
                #print(X_features[j])
                #print(df['Type'][j])
                if(df['Type'][j] == k+1):
                    count = count + 1
    j_dist = count / X_features.shape[0]
    print("Jaccard Distance for " + str(k+1) + " cluster is: " + str(j_dist) )


