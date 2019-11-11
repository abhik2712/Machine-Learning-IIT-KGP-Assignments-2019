#ROLL             : 17CS30001
#NAME             : ABHIK NASKAR
#ASSIGNMENT NUMBER: 03
#Compile by using python3 17CS30001_3.py on terminal




import numpy as np
import math
import pandas as pd

class DecisionTreeStump():
    def __init__(self):
 
        self.feature_index = None
        
        #the unique_elem value that the feature should be measured against
        self.unique_elem = None
        
        #value indicative of the classifier's accuracy
        self.alpha = None
        
class Adaboost():

    def __init__(self, n_clf = 3):
             self.n_clf = n_clf
    
    def fit(self, X, y):
            n_samples = np.shape(X)[0]
            n_features = np.shape(X)[1]

            #initialise weights to 1/N
            
            weights = np.full(n_samples, (1 / n_samples))
            
            self.clfs = []
            #iterate through classifiers
            for _ in range(self.n_clf):
                clf = DecisionTreeStump()
               
                min_error = 2
                
                for feature_i in range(n_features):
                    
                    feature_values = np.expand_dims(X[:, feature_i], axis = 1)
                   
                    unique_values = np.unique(feature_values)
                    
                    for unique_elem in unique_values:
                        
                        prediction = np.ones(np.shape(y))
                        
                        prediction[X[:, feature_i] < unique_elem] = -1
                        
                        error = sum(weights[y != prediction])
                        
                        if error > 0.5:
                            error = 1 - error
                            
                        if error < min_error:
                            clf.unique_elem = unique_elem
                            clf.feature_index = feature_i
                            min_error = error
                
               
                shape_y = np.shape(y)
                predictions = np.ones(shape_y)
                
                clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))

                predictions[(X[:, clf.feature_index] < clf.unique_elem)] = -1

                weights *= np.exp(-clf.alpha * y * predictions)
               
                weights /= np.sum(weights)
                
                self.clfs.append(clf)
                
    def predict(self, X):

        n = np.shape(X)

        n_samples = n[0]
        y_pred = np.zeros((n_samples, 1))
       
        for clf in self.clfs:
            
            shape_y_pred = np.shape(y_pred)
            predictions = np.ones(shape_y_pred)
            
            predictions[(X[:, clf.feature_index] < clf.unique_elem)] = -1
           
            y_pred += clf.alpha * predictions

        y_pred = np.sign(y_pred).flatten()
        
        return y_pred

    
    
if __name__ == "__main__":
    
    df_train = pd.read_csv('/Users/abhiknaskar/Desktop/Machine Learning IIT KGP/Assignment 3/data3_19.csv')
    df_test = pd.read_csv('/Users/abhiknaskar/Desktop/Machine Learning IIT KGP/Assignment 3/test3_19.csv')
    
    df_train.pclass[df_train.pclass == '1st'] = 1
    df_train.pclass[df_train.pclass == '2nd'] = 2
    df_train.pclass[df_train.pclass == '3rd'] = 3
    df_train.pclass[df_train.pclass == 'crew'] = 4
    
    df_test.pclass[df_test.pclass == '1st'] = 1
    df_test.pclass[df_test.pclass == '2nd'] = 2
    df_test.pclass[df_test.pclass == '3rd'] = 3
    df_test.pclass[df_test.pclass == 'crew'] = 4
    
    
    df_train.age[df_train.age == 'adult'] = 1
    df_train.age[df_train.age == 'child'] = 2
    
    df_test.age[df_test.age == 'adult'] = 1
    df_test.age[df_test.age == 'child'] = 2
    
    df_train.gender[df_train.gender == 'male'] = 1
    df_train.gender[df_train.gender == 'female'] = 2
    
    df_test.gender[df_test.gender == 'male'] = 1
    df_test.gender[df_test.gender == 'female'] = 2
    
    df_train.survived[df_train.survived == 'yes'] = 1
    df_train.survived[df_train.survived == 'no'] = -1
    
    df_test.survived[df_test.survived == 'yes'] = 1
    df_test.survived[df_test.survived == 'no'] = -1
    
    
    y_list_train = df_train['survived'].tolist()
    y_train = np.asarray(y_list_train)
    #print(y_train)
    
    df_train = df_train.drop(['survived'], axis = 1)
   # print(df_train.head(5))
    
    X_train = df_train.rename_axis('pclass').values
    
    
    #print(X_train)
    #print(np.shape(X_train))
        
    y_list_test = df_test['survived'].tolist()
    y_test = np.asarray(y_list_test)
    
    #print(y_test)
    
    df_test = df_test.drop(['survived'], axis = 1)
    
    X_test = df_test.rename_axis('pclass').values
    #print(X_test)
    
    clf = Adaboost(n_clf = 3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    count = 0
    
    for i in range(0, len(y_test)):
        if(y_test[i] == y_pred[i]):
            count = count + 1
    
    accuracy = 100 * count / float(len(y_test))
    
    print("Accuracy: " + str(accuracy))

