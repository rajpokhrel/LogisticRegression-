import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib as plt
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier

dataArray = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

with open('data/bank-additional-full.csv') as bankFile:
    bankReader = csv.reader(bankFile, delimiter=';')
    next(bankReader)
    for row in bankReader:
        for i in range(21):
            dataArray[i].append(row[i])

dataK1 = pickle.load(open("data/feature_selection/feature_k=1.pkl",'rb'))
dataK3 = pickle.load(open("data/feature_selection/feature_k=3.pkl",'rb'))
dataK5 = pickle.load(open("data/feature_selection/feature_k=5.pkl",'rb'))


Num1 = dataK1['numerical']
Cat1 = dataK1['categorical']
Num3 = dataK3['numerical']
Cat3 = dataK3['categorical']
Num5 = dataK5['numerical']
Cat5 = dataK5['categorical']

Xc = dataK3['categorical'][:,:11]
Xn = dataK5['numerical'][:,:5]
X = np.concatenate((Xc,Xn),axis=1)

X1 = np.concatenate((Num1,Cat1),axis=1)
X3 = np.concatenate((Num3,Cat3),axis=1)
X5 = np.concatenate((Num5,Cat5),axis=1)
y = []

for i in range(41188):
    y.append(dataK3['categorical'][i][12])

#Convert list to np array
y = np.asarray(y)

kf = KFold(n_splits = 5, random_state= 42)
for train_indices, test_indices in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=500)
    mlp.fit(X, y)
    predictions = mlp.predict(X_test)
    accur = accuracy_score(y_true=y_test, y_pred=predictions, normalize=True, sample_weight=None)
    f1score = f1_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    precision = precision_score(y_test, predictions, average='macro')
    
    print("Normalize Values")
    print('Accuracy Score: ',accur)
    print('Recall:', recall)
    print('F1-Score:', f1score)
    print('Precision:', precision)
    print(predictions)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))


from sklearn import tree
kf = KFold(n_splits = 5, random_state= 42)
for train_indices, test_indices in kf.split(X):
    dt = tree.DecisionTreeClassifier(criterion='gini',splitter='best')
    dt.fit(X,y)
    predictions_d = dt.predict(X_test)
    accur_d = accuracy_score(y_true=y_test, y_pred=predictions_d, normalize=True, sample_weight=None)
    f1score_d = f1_score(y_test, predictions_d, average='macro')
    recall_d = recall_score(y_test, predictions_d, average='macro')
    precision_d = precision_score(y_test, predictions_d, average='macro')
    print('Accuracy Score: ',accur_d)
    print('Recall:', recall_d)
    print('F1-Score:', f1score_d)
    print('Precision:', precision_d)
    print(predictions_d)
    print(confusion_matrix(y_test,predictions_d))
    print(classification_report(y_test,predictions_d))


'''
    kf = KFold(n_splits = 5, random_state= 42)
    for train_indices, test_indices in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X=X_train, y=y_train)
    clf.feature_importances_
    clf.score(X=X_test, y=y_test)
    predictions = clf.predict(X_test)
    print(predictions)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    '''

'''
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    xDataArray = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    with open('data/bank-additional-full.csv') as bankFile2:
    bankReader2 = csv.reader(bankFile2, delimiter=';')
    next(bankReader2)
    for row in bankReader2:
    xDataArray[0].append(row[0])
    xDataArray[1].append(row[1])
    xDataArray[2].append(row[2])
    xDataArray[3].append(row[3])
    xDataArray[4].append(row[4])
    xDataArray[5].append(row[5])
    xDataArray[6].append(row[6])
    xDataArray[7].append(row[7])
    xDataArray[8].append(row[8])
    xDataArray[9].append(row[9])
    xDataArray[10].append(row[10])
    xDataArray[11].append(row[11])
    xDataArray[12].append(row[12])
    xDataArray[13].append(row[13])
    xDataArray[14].append(row[14])
    xDataArray[15].append(row[15])
    xDataArray[16].append(row[16])
    xDataArray[17].append(row[17])
    xDataArray[18].append(row[18])
    xDataArray[19].append(row[19])
    xDataArray[20].append(row[20])
    
    df = pd.DataFrame({'age': xDataArray[0],
    'job': xDataArray[1],
    'marital': xDataArray[2],
    'edcation':xDataArray[3],
    'default':xDataArray[4],
    'housing':xDataArray[5],
    'loan':xDataArray[6],
    'contact':xDataArray[7],
    'month':xDataArray[8],
    'day_of_week':xDataArray[9],
    'duration': xDataArray[10],
    'campign':xDataArray[11],
    'pdays':xDataArray[12],
    'previous':xDataArray[13],
    'poutcome':xDataArray[14],
    'emp.var.rate':xDataArray[15],
    'cons.price.idx':xDataArray[16],
    'cons.conf.idx':xDataArray[17],
    'euribor3m':xDataArray[18],
    'nr.employed':xDataArray[19],
    'y':xDataArray[20] })
    
    
    df.apply(LabelEncoder().fit_transform)
    array = df.values
    
    #numpy.array(array)
    X = array[:,0:19]
    Y = array[:,20]
    
    encodeA = LabelEncoder()
    
    job_n = encodeA.fit_transform(xDataArray[1])
    marital_n = encodeA.fit_transform(xDataArray[2])
    education_n = encodeA.fit_transform(xDataArray[3])
    default_n = encodeA.fit_transform(xDataArray[4])
    housing_n = encodeA.fit_transform(xDataArray[5])
    loan_n = encodeA.fit_transform(xDataArray[6])
    contact_n = encodeA.fit_transform(xDataArray[7])
    month_n = encodeA.fit_transform(xDataArray[8])
    day_n= encodeA.fit_transform(xDataArray[9])
    poutcome_n = encodeA.fit_transform(xDataArray[14])
    
    y = encodeA.fit_transform(xDataArray[20])
    
    age_n = np.array(xDataArray[0]).reshape(-1,1)
    job_n = np.array(job_n).reshape(-1,1)
    marital_n = np.array(marital_n).reshape(-1,1)
    education_n = np.array(education_n).reshape(-1,1)
    default_n = np.array(default_n).reshape(-1,1)
    housing_n = np.array(housing_n).reshape(-1,1)
    loan_n = np.array(loan_n).reshape(-1,1)
    conact_n = np.array(contact_n).reshape(-1,1)
    month_n = np.array(month_n).reshape(-1,1)
    day_n = np.array(day_n).reshape(-1,1)
    duration_n = np.array(xDataArray[10]).reshape(-1,1)
    campaign_n = np.array(xDataArray[11]).reshape(-1,1)
    pdays_n = np.array(xDataArray[12]).reshape(-1,1)
    previous_n = np.array(xDataArray[13]).reshape(-1,1)
    poutcome_n = np.array(poutcome_n).reshape(-1,1)
    empVarRate_n = np.array(xDataArray[15]).reshape(-1,1)
    consPriceIndex_n = np.array(xDataArray[16]).reshape(-1,1)
    consConfIndex_n = np.array(xDataArray[17]).reshape(-1,1)
    euribor3m_n = np.array(xDataArray[18]).reshape(-1,1)
    nrEmployed_n = np.array(xDataArray[19]).reshape(-1,1)
    
    outputX = np.concatenate((job_n, marital_n, education_n, default_n, housing_n, loan_n, conact_n, month_n, day_n, poutcome_n), axis=1)
    outputY = np.concatenate((age_n, campaign_n, pdays_n, previous_n, empVarRate_n, consPriceIndex_n, consConfIndex_n, euribor3m_n, nrEmployed_n),axis=1)
    
    kf = KFold(n_splits = 5, random_state= 10)
    for train_indices, test_indices in kf.split(outputX):
    dt = tree.DecisionTreeClassifier(criterion='gini',splitter='best')
    dt.fit(outputX,y)
    predictions_d = dt.predict(X_test)
    accur_d = accuracy_score(y_true=y_test, y_pred=predictions_d, normalize=True, sample_weight=None)
    f1score_d = f1_score(y_test, predictions_d, average='macro')
    recall_d = recall_score(y_test, predictions_d, average='macro')
    precision_d = precision_score(y_test, predictions_d, average='macro')
    print('Accuracy Score: ',accur_d)
    print('Recall:', recall_d)
    print('F1-Score:', f1score_d)
    print('Precision:', precision_d)
    print(predictions_d)
    print(confusion_matrix(y_test,predictions_d))
    print(classification_report(y_test,predictions_d))
    '''
