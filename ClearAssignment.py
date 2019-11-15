# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:58:13 2019

@author: Pigi
"""

#clear everything up

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn import datasets
from sklearn import preprocessing

#connecting to files
train_data = pd.read_excel(r"C:\Users\Pigi\Desktop\Erasmus Semester Lessons\Data Mining\Project Assignment -AXA\Data_DSC2019_STUDENTS\DSC2019_Training.xlsx",  sheet_name='Sheet1')
test_data= pd.read_excel(r"C:\Users\Pigi\Desktop\Erasmus Semester Lessons\Data Mining\Project Assignment -AXA\Data_DSC2019_STUDENTS\DSC2019_Test.xlsx",  sheet_name='Sheet1')


#finding out how many nulls does each column has
train_data.isnull().sum()

#dropping all columns with missing value percentage > 50%
perc_mv= (train_data.isnull().sum())/len(train_data)*100
for i in train_data:
    if perc_mv[i]>50: 
       print("'{i}' need to be dropped".format(i=i))
       train_data.drop([i], axis=1, inplace= True)

#distinguishing columns
ID= train_data.iloc[:,0]
target_var= train_data.iloc[:,-1]

#searching for all categorical columns ('object') and adding them to a list
categorical_cols= []
for i in train_data:
    if train_data[i].dtypes == 'object':
        print("'{i}' is going to the categorical values".format(i=i))
        categorical_cols.append(i)

#creating a list with all the numerical columns 
numerical_cols= list(set(list(train_data.columns))-set(categorical_cols)-set(ID)-set(target_var))

#imputing the mean to missing values of the numerical columns
imp1 = SimpleImputer(strategy="mean")
train_data[numerical_cols]=imp1.fit_transform(train_data[numerical_cols])    
 
#will do the same for categorical columns (without the if statement)
#should I normalize the column LoanID???
for col in train_data: 
    a= False
    for i in numerical_cols:
        if col==i:
            a= True
            train_data[col]= (train_data[col]- train_data[col].min())/(train_data[col].max()- train_data[col].min())

#imputing the most frequent value to missing values of the categorical columns
imp2 = SimpleImputer(strategy="most_frequent")
train_data[categorical_cols]=imp2.fit_transform(train_data[categorical_cols])     

# Create a set of dummy variables for the columns that have less than 3 categories and then drop them
#DO I REALLY NEED 3? FOR WHICH ONES AM I GONNA CREATE DUMMIES?
#WHAT AM I GONNA DO WITH THE REST?
for col_name in train_data.columns:
    if train_data[col_name].dtypes== 'object':
        unique= len(train_data[col_name].unique())
        if unique <= 3: 
            dummies = pd.get_dummies(train_data[col_name])
            train_data = pd.concat([train_data, dummies], axis=1)
            train_data.drop(col_name,axis=1, inplace= True)

print(train_data.isnull().any())

#separating the train and the test data
Y= train_data['Label_Default'].values
X= train_data.drop(labels='Label_Default',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
'''from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, Y_train)'''