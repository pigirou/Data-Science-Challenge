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

#distinguishing column values
ID= train_data.iloc[:,0]
target_var= train_data.iloc[:,-1]

#seperating the name of the target variable and the LoanID variable
ID_name= train_data.columns[0]
target_name= train_data.columns[-1]


#searching for all categorical columns ('object') and adding them to a list
categorical_cols= []
for i in train_data:
    if train_data[i].dtypes == 'object':
        if i!=target_name:
            print("'{i}' is going to the categorical values".format(i=i))
            categorical_cols.append(i)


#creating a list with all the numerical columns 
numerical_cols= list(train_data.columns)
for i in train_data.columns:
    if i in categorical_cols:
        numerical_cols.remove(i)
    if i== target_name:
        numerical_cols.remove(i)   
    if i== ID_name:
        numerical_cols.remove(i)   

#imputing the mean to missing values of the numerical columns
imp1 = SimpleImputer(strategy="mean")
train_data[numerical_cols]=imp1.fit_transform(train_data[numerical_cols])    
 
#normalizing all numerical data
for col in train_data: 
    for i in numerical_cols:
        if i==col:
            train_data[i]= (train_data[i]- train_data[i].min())/(train_data[i].max()- train_data[i].min())

#imputing the most frequent value to missing values of the categorical columns
imp2 = SimpleImputer(strategy="most_frequent")
train_data[categorical_cols]=imp2.fit_transform(train_data[categorical_cols])     

# Create a set of dummy variables for the categorical features
for col_name in categorical_cols:
    #top_10= the top 10 (most frequent) values for each categorical column
    top_10=[x for x in train_data[col_name].value_counts().sort_values(ascending= False).head(10).index]
    for categ in top_10:
        #creating dummies for only the ten most frequent values
        train_data[categ]= np.where(train_data[col_name]== categ, 1, 0)
    train_data[[col_name]+ top_10].head(40)
    train_data.drop(col_name, axis=1, inplace= True)

print(train_data)

#separating the train and the test data
Y= train_data['Label_Default'].values
X= train_data.drop(labels='Label_Default',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
'''from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, Y_train)'''