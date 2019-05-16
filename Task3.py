import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib as plt
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.metrics import classification_report,confusion_matrix

#load the pickle file
dataK1 = pickle.load(open("data/feature_selection/feature_k=1.pkl",'rb'))

print(dataK1['categorical'].shape)
print(dataK1['numerical'].shape)

#X = np.concatenate((dataK1['categorical'],dataK1['numerical']),axis=1)
X = np.array(dataK1['numerical'])
y = []
for i in range(41188):
    y.append(dataK1['categorical'][i][2])

#Convert list to np array
y = np.asarray(y)

#Initialize K-fold
kf = KFold(n_splits = 5, random_state= 5)

#K1 model
for train_indices, test_indices in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
    lg = LogisticRegression()
    lg = lg.fit(X[train_indices], y[train_indices])
    y_pred_test = lg.predict(X_test)
    
    CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
    acs = accuracy_score(y_true=y_test, y_pred=y_pred_test, normalize=True, sample_weight=None)
    f1score = f1_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')
    precision = precision_score(y_test, y_pred_test, average='macro')
    kfold = cross_val_score(estimator = lg, X = X[train_indices], y = y[train_indices], cv = 5)
    
    print('Confusion Matrix: \n',CM)
    print('Accuracy Score: ',acs)
    print('Recall:', recall)
    print('F1-Score:', f1score)
    print('Precision:', precision)
    print("5-Fold Cross Vaildation:", kfold.mean())
    print('\n')
    print(confusion_matrix(y_test,y_pred_test))
    print(classification_report(y_test,y_pred_test))


dataK3 = pickle.load(open("data/feature_selection/feature_k=3.pkl",'rb'))

Xc = dataK3['categorical'][:,:11]
Xn = dataK3['numerical'][:,:5]
X = np.concatenate((Xc,Xn),axis=1)

y = []
for i in range(41188):
    y.append(dataK3['categorical'][i][12])

#Convert list to np array
y = np.asarray(y)

#Initialize K-fold
kf = KFold(n_splits = 5, random_state= 20)

#K1 model
for train_indices, test_indices in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    lg = LogisticRegression()
    lg = lg.fit(X[train_indices], y[train_indices])
    y_pred_test = lg.predict(X_test)
    
    CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
    acs = accuracy_score(y_true=y_test, y_pred=y_pred_test, normalize=True, sample_weight=None)
    f1score = f1_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')
    precision = precision_score(y_test, y_pred_test, average='macro')
    kfold = cross_val_score(estimator = lg, X = X[train_indices], y = y[train_indices], cv = 5)
    
    print("Normalize Values")
    print('Confusion Matrix: \n',CM)
    print('Accuracy Score: ',acs)
    print('Recall:', recall)
    print('F1-Score:', f1score)
    print('Precision:', precision)
    print("5-Fold Cross Vaildation:", kfold.mean())
    print('\n')
    print(confusion_matrix(y_test,y_pred_test))
    print(classification_report(y_test,y_pred_test))


dataK5 = pickle.load(open("data/feature_selection/feature_k=5.pkl",'rb'))
Num5 = dataK5['numerical'][:,:27]
Cat5 = dataK5['categorical'][:,:5]

print(dataK5['categorical'].shape)
print(dataK5['numerical'].shape)


X = np.concatenate((Num5,Cat5),axis=1)

y = []
for i in range(41188):
    y.append(dataK5['categorical'][i][26])

y = np.asarray(y)
kf = KFold(n_splits = 5, random_state= 35)
for train_indices, test_indices in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    lg = LogisticRegression()
    lg = lg.fit(X[train_indices], y[train_indices])
    y_pred_test = lg.predict(X_test)
    
    CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
    acs = accuracy_score(y_true=y_test, y_pred=y_pred_test, normalize=True, sample_weight=None)
    f1score = f1_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')
    precision = precision_score(y_test, y_pred_test, average='macro')
    kfold = cross_val_score(estimator = lg, X = X[train_indices], y = y[train_indices], cv = 5)
    
    print("Normalize Values")
    print('Confusion Matrix: \n',CM)
    print('Accuracy Score: ',acs)
    print('Recall:', recall)
    print('F1-Score:', f1score)
    print('Precision:', precision)
    print("5-Fold Cross Vaildation:", kfold.mean())
    print(confusion_matrix(y_test,y_pred_test))
    print(classification_report(y_test,y_pred_test))
    print('\n')


