# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:26:03 2019

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

def encoding(df):

    #Seperate the name of the target variable and the LoanID variable
    ID_name= df.columns[0]
    target_name= df.columns[-1]

        
   #Search for all categorical columns and add them to a list
    
    categorical_cols= []
    for i in df:
        if i!=target_name:
            if df[i].dtypes == 'object':
                categorical_cols.append(i)
                
    for col_name in categorical_cols:
        unique= len(train_data[col_name].unique())
        print("'{i}' has '{u}' unique categories".format(i=col_name, u=unique))
     
    #Impute the most frequent value to missing values of the categorical columns
    imp2 = SimpleImputer(strategy="most_frequent")
    df[categorical_cols]=imp2.fit_transform(df[categorical_cols])
    
    cat_columns= []
    for i in df:
        if i!=target_name:
            if df[i].dtypes == 'object':
                df[i]=df[i].astype('category')
                cat_columns.append(i)  
                
                
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    
      
   
    #Create a list with all the numerical columns 
    numerical_cols= list(df.columns)
    for i in df.columns:
        if i in categorical_cols:
            numerical_cols.remove(i)
        if i== target_name:
            numerical_cols.remove(i)   
        if i== ID_name:
            numerical_cols.remove(i)   

    #Impute the mean to missing values of the numerical columns
    imp1 = SimpleImputer(strategy="mean")
    df[numerical_cols]=imp1.fit_transform(df[numerical_cols])  
       
     
    #Normalize all numerical data
    for col in df: 
        for i in numerical_cols:
            if i==col:
                df[i]= (df[i]- df[i].min())/(df[i].max()- df[i].min())
                
    for col in df: 
        for i in categorical_cols:
            if i==col:
                df[i]= (df[i]- df[i].min())/(df[i].max()- df[i].min())
    
   
   
    #Create correlation matrix
    corr_matrix = df.corr().abs()
    
    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    #Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print("High correlation columns")
    print(to_drop)
    

    rs = np.random.RandomState(0)
    corr_df = pd.DataFrame(rs.rand(10, 10))
    correlation = corr_df.corr()
    correlation.style.background_gradient(cmap='coolwarm')
    print(correlation)
    return df

def evaluate_classifier(X_train, X_test, y_train, y_test):
    
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, roc_auc_score
    from sklearn import metrics


    # Test the random forest classifier
    classifier = RandomForestClassifier(n_estimators=1000)
    classifier.fit(X_train, y_train)
    model = classifier.fit(X_train, y_train)
    
    X1= test_data.drop(labels='Label_Default',axis=1)
    
    result = model.predict(X1)
    result_proba = model.predict_proba(X1)[:,1]
    
    new_frame = []
    new_frame = pd.DataFrame(new_frame)
    new_frame['LoanID']= test_data.iloc[:,0]
    new_frame['Predictions']= result_proba
    
    counter=0
    c=0
    for i in result:
        if i=='Y':
            counter=counter+1
        else: 
            c=c+1
    
    perc= counter/len(result)*100
    print("Percentage of default '{perc}'".format(perc=perc))
    perc1= c/len(result)*100
    print("Percentage of non-default '{perc}'".format(perc=perc1))
    
    print("Probabilities - results")
    for i in range(len(X1)):
        print("'{i1}'   '{y}'".format(i1=result[i], y = result_proba[i]))
    print("New frame")
    print(new_frame)
    
    score = f1_score(y_test, classifier.predict(X_test), pos_label='Y')
    
    # Generate the P-R curve
    y_prob = classifier.predict_proba(X_test)
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1], pos_label='Y')
    
    # Include the score in the title
    yield 'Random forest (F1 score={:.3f})'.format(score), precision, recall
    
    # Generate the ROC curve for random forest
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_prob[:,1], pos_label='Y')
    auc = roc_auc_score(y_test, y_prob[:,1])
    print('Random forest AUC: %.3f' % auc)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='Random Forest RT + LR')
  
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')


    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    model = classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test), pos_label='Y')
    
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    print(y_prob)
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label='Y')
    
    # Generate the ROC curve for Ada
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_prob, pos_label='Y')
    auc = roc_auc_score(y_test, y_prob)
    print('Ada boost AUC: %.3f' % auc)

    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='Ada boost RT + LR')
   
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')


    
    #==============================================
          
def plot(results):

    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ')

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    plt.tight_layout()

    plt.show()
    plt.close()


# =====================================================================
if __name__ == '__main__':

    train_data = pd.read_excel(r"C:\Users\Pigi\Desktop\Erasmus Semester Lessons\Data Mining\Project Assignment -AXA\Data_DSC2019_STUDENTS\DSC2019_Training.xlsx",  sheet_name='Sheet1')
    test_data= pd.read_excel(r"C:\Users\Pigi\Desktop\Erasmus Semester Lessons\Data Mining\Project Assignment -AXA\Data_DSC2019_STUDENTS\DSC2019_Test.xlsx",  sheet_name='Sheet1')
         
    target_name= train_data.columns[-1]
    
    #Encode the test and train set
    train_data= encoding(train_data)
    test_data= encoding(test_data)
    X2= test_data.drop(labels='Label_Default',axis=1)
    

    print("target column: '{i}'".format(i=train_data[target_name]))

    
    #Separate the train and the test data
    Y= train_data['Label_Default'].values
    X= train_data.drop(labels='Label_Default',axis=1)
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
  
    
    # Evaluate classifiers on the data
    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))
    print(results)
    
    # Display the results
    print("Plotting the results")
    plot(results)