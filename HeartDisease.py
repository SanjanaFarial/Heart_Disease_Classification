# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:59:54 2018

@author: taarn
"""

import numpy as np 
import pandas as pd 

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import sem 



#Reading the dataset
dataset = pd.read_csv('heart_disease.csv')

#Analyinge Data
dataset.head()

#converting strings into numerical values
dataset.replace({'positive': 1, 'negative': 0, 
                 'yes':1, 'no':0, 
                 'normal':0, 'left_vent_hyper':1, 'st_t_wave_abnormality':2,
                 't':1, 'f':0,
                 'asympt':1, 'atyp_angina':2, 'non_anginal':3, 'typ_angina':4,
                 'female':0, 'male':1}, inplace=True)

#saving the new values into another file
dataset.to_csv('heart_numerical.csv', index=False)



#defining features and target value
target = dataset['disease']
features = dataset.drop('disease', axis = 1)
#target = np.array(target)
#features = np.array(features)



#train test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.20, random_state=37)



#scaler transform
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



cv = KFold(features.shape[0], 5, shuffle=True, random_state=33)



def classReport(y_train, y_train_pred, y_pred, y_test):    
    #metrics creation
    print("\nMatrix accuracy score for training")
    print(metrics.accuracy_score(y_train, y_train_pred))
    print("\nMatrix accuracy score for testing") 
    print(metrics.accuracy_score(y_pred, y_test))
    print('\n')  
    print(metrics.classification_report(y_test, y_pred))
    metrics.confusion_matrix(y_test, y_pred)



def mean_score(scores):
    return ("\nAccuracy: {0:.3f} (+/-{1:.3f})\n").format(np.mean(scores), sem(scores))

def print_score(score):
    print(score)
    print(mean_score(score))



######################################################################################
#using SGD classifier
print('\nResult of SGD')

#classifer and fit
clf1 = Pipeline([('scaler', StandardScaler()),('', SGDClassifier())])
clf1.fit(X_train, y_train)

#prediction
y_train_pred1 = clf1.predict(X_train)
y_pred1 = clf1.predict(X_test)

#metrics creation
classReport(y_train, y_train_pred1, y_pred1, y_test)


#print score
scores1 = cross_val_score(clf1, features, target, cv=cv)
mean_score(scores1)
print_score(scores1)



#######################################################################################
#Using SVM classifier 
print('\nResult of SVM')
#classifer and fit
clf2 = Pipeline([('scaler', StandardScaler()),('',svm.SVC())])
clf2.fit(X_train, y_train)

#prediction
y_train_pred2 = clf2.predict(X_train)
y_pred2 = clf2.predict(X_test)

#metrics creation
classReport(y_train, y_train_pred2, y_pred2, y_test)

#Cross fit metrics 
scores2 = cross_val_score(clf2, features, target, cv=cv) 
mean_score(scores2)
print_score(scores2)




####################################################################################
#using knn classifier
print('\nResult of KNN Classifier')

# fitting the model
clf3 = Pipeline([('scaler', StandardScaler()),('',KNeighborsClassifier(n_neighbors=3))])
clf3.fit(X_train, y_train)

# predict the response
y_train_pred3 = clf3.predict(X_train)
y_pred3 = clf3.predict(X_test)

#metrics creation
classReport(y_train, y_train_pred3, y_pred3, y_test)

# perform 5-fold cross validation
scores3 = cross_val_score(clf3, features, target, cv=cv)
mean_score(scores3)
print_score(scores3)
  

 

###################################################################################### 
#using random forest
print('\nResult of Random Forest Classifier') 
#classifier and fit
#clf4 = RandomForestClassifier(n_estimators=100)
clf4 = Pipeline([('scaler', StandardScaler()),('',RandomForestClassifier(n_estimators=100))])
clf4.fit(X_train, y_train)

#prediction       
y_train_pred4 = clf4.predict(X_train)
y_pred4 = clf4.predict(X_test)

#metrics creation
classReport(y_train, y_train_pred4, y_pred4, y_test)

#Cross fit metrics 
scores4 = cross_val_score(clf4, features, target, cv=cv) 
mean_score(scores4)
print_score(scores4)





######################################################################################
#using bagging
print('\nResult of Bagging Classifier') 
#classifier and fit
#clf5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), max_features=0.5)
clf5 = Pipeline([('scaler', StandardScaler()),('',BaggingClassifier(base_estimator=DecisionTreeClassifier(), max_samples= 0.5, max_features=1.0, n_estimators= 20))])
clf5 = clf5.fit(X_train, y_train)

#prediction 
y_train_pred5 = clf5.predict(X_train)
y_pred5 = clf5.predict(X_test)

#metrics creation
classReport(y_train, y_train_pred5, y_pred5, y_test)

#Cross fit metrics 
scores5 = cross_val_score(clf5, features, target, cv=cv) 
mean_score(scores5)
print_score(scores5)



