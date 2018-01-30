# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:05:01 2017

@author: Nafis
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_recall_fscore_support as prfs


dataset =pd.read_csv('Churn_Modelling.csv')
df=dataset.iloc[:,3:13]
df=pd.get_dummies(df)
df=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,12]]
X=df.values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_scaled=scale.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense

model1=Sequential()

model1.add(Dense(6,activation='relu',input_dim=11))
model1.add(Dense(6,activation='relu'))
model1.add(Dense(1,activation='sigmoid',input_dim=11))

model1.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])

model1.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred=model1.predict(X_test)
y_pred=(y_pred>0.5)

print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))

#logistic Regression
from sklearn.linear_model import LogisticRegression as lr
model2=lr().fit(X_train,y_train)
y_pred=model2.predict(X_test)
y_pred=(y_pred>0.5)
print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))

from sklearn.neighbors import KNeighborsClassifier as knn
model3=knn().fit(X_train,y_train)
y_pred=model3.predict(X_test)
print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))

from sklearn.svm import SVC
model4=SVC().fit(X_train,y_train)
y_pred=model4.predict(X_test)
print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))

from sklearn.naive_bayes import GaussianNB
model5=GaussianNB().fit(X_train,y_train)
y_pred=model5.predict(X_test)
print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier as tree
model6=tree(criterion='entropy').fit(X_train,y_train)
y_pred=model6.predict(X_test)
print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier as forest
model7=forest(max_depth=5).fit(X_train,y_train)
y_pred=model7.predict(X_test)
print('CM:',confusion_matrix(y_test,y_pred))
print('AC:',ac(y_test,y_pred))
print('F1 scores:',f1(y_test,y_pred))
print('PR:',prfs(y_test,y_pred))