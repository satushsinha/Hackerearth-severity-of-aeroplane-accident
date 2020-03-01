# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 00:16:01 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('train2.csv')
#dataset_test = pd.read_csv('test.csv')


x_train=dataset.iloc[:,1:9]
y_train=dataset.iloc[:,0]
#x_train=x_train.iloc[:,[0,7]]

y_train=pd.DataFrame(y_train)

#x_train['Total_Safety_Complaints'] = np.power(3, x_train['Total_Safety_Complaints'])
x_train['Days_Since_Inspection'] = np.power(2.5, x_train['Days_Since_Inspection'])
x_train['Cabin_Temperature'] = np.power(3.5, x_train['Cabin_Temperature'])
#x_train['Violations'] = np.power(3, x_train['Violations'])
x_train['Turbulence_In_gforces'] = np.power(3, x_train['Turbulence_In_gforces'])

x_train['Max_Elevation'] = np.log2(x_train['Max_Elevation'])
#x_train['Safety_Score'] = np.log(x_train['Safety_Score'])

'''
dataset_test['Total_Safety_Complaints'] = np.power(3, dataset_test['Total_Safety_Complaints'])
dataset_test['Days_Since_Inspection'] = np.power(2.5, dataset_test['Days_Since_Inspection'])
dataset_test['Cabin_Temperature'] = np.power(3.5, dataset_test['Cabin_Temperature'])
dataset_test['Violations'] = np.power(3, dataset_test['Violations'])
dataset_test['Turbulence_In_gforces'] = np.power(3,dataset_test['Turbulence_In_gforces'])

dataset_test['Max_Elevation'] = np.log2(dataset_test['Max_Elevation'])

X_test=dataset_test.iloc[:,0:9]
'''

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
y_train.iloc[:, 0] = labelencoder_X.fit_transform(y_train.iloc[:, 0])
'''
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()
'''


        
#y_pred3=pd.DataFrame(y_pred3)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)
'''
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 5, nb_epoch = 50)
'''

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =1000, criterion = 'entropy', max_depth=50, min_samples_split=3,min_samples_leaf=1,)
classifier.fit(X_train, y_train)

'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
'''


y_pred = classifier.predict(X_test)


#y_pred= (y_pred > 0.5)
'''
#df[['COL2', 'COL4']] = (df[['COL2', 'COL4']] == 'TRUE').astype(int)

#y_pred=y_pred.astype(int)


y_pred2=[0]*2500
for i in range(0,2500):
    d=0
    for j in range(0,4):
        if(y_pred[i,j]>d):
            d=y_pred[i,j]
            y_pred2[i]=j
            
 
#y_pred2=pd.DataFrame(y_pred2)     
       
y_pred3=[]
for i in range(0,2500):
    for j in range(0,4):
        if(y_test[i,j] ==1):
            y_pred3.append(j)
        
y_pred3=pd.DataFrame(y_pred3)
     
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''''''''
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
# Sequential Forward Selection(sfs)
sfs = SFS(RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0),
           k_features=8,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)

sfs.fit(X_train, y_train)
z=sfs.k_feature_names_

''''''''''''''''''
max_depth=100, min_samples_split=5,min_samples_leaf=1
'''''''''