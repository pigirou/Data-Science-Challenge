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
from pandas import Series,DataFrame
from pylab import rcParams
import seaborn as sb
import scipy 



def encoding(df):

    #seperating the name of the target variable and the LoanID variable
    ID_name= df.columns[0]
    target_name= df.columns[-1]
    
    #dropping all columns with missing value percentage > 50%
    perc_mv= (df.isnull().sum())/len(df)*100
    for i in df:
        if perc_mv[i]>80: 
           df.drop([i], axis=1, inplace= True)


    #searching for all categorical columns ('object') and adding them to a list
    categorical_cols= []
    for i in df:
        if i!=target_name:
            if df[i].dtypes == 'object':
                categorical_cols.append(i)


    #creating a list with all the numerical columns 
    numerical_cols= list(df.columns)
    for i in df.columns:
        if i in categorical_cols:
            numerical_cols.remove(i)
        if i== target_name:
            numerical_cols.remove(i)   
        if i== ID_name:
            numerical_cols.remove(i)   

    #imputing the mean to missing values of the numerical columns
    imp1 = SimpleImputer(strategy="mean")
    df[numerical_cols]=imp1.fit_transform(df[numerical_cols])    
     
    #normalizing all numerical data
    for col in df: 
        for i in numerical_cols:
            if i==col:
                df[i]= (df[i]- df[i].min())/(df[i].max()- df[i].min())

    #imputing the most frequent value to missing values of the categorical columns
    imp2 = SimpleImputer(strategy="most_frequent")
    df[categorical_cols]=imp2.fit_transform(df[categorical_cols])     

    # Create a set of dummy variables for the categorical features
    for col_name in categorical_cols:
        #top_10= the top 10 (most frequent) values for each categorical column
        top_10=[x for x in df[col_name].value_counts().sort_values(ascending= False).head(10).index]
        for categ in top_10:
            #creating dummies for only the ten most frequent values
            df[categ]= np.where(df[col_name]== categ, 1, 0)
        df[[col_name]+ top_10].head(40)
        df.drop(col_name, axis=1, inplace= True)
   
    # create correlation matrix
    corr_matrix = df.corr().abs()
    # select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Drop features 
    df.drop(df[to_drop], axis=1)
    return df

def evaluate_classifier(X_train, X_test, y_train, y_test):

#Import some classifiers to test from sklearn.svm 
    
    #from sklearn.svm import LinearSVC, NuSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier

    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, roc_auc_score
    from sklearn import metrics
    
    
    # testing the random forest classifier
    classifier = RandomForestClassifier(n_estimators=1000)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    model = classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test), pos_label=1)
    
    # Generate the P-R curve
    
    #y_prob = classifier.decision_function(X_test)
    y_prob = classifier.predict_proba(X_test)
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1], pos_label=1)
    # Include the score in the title
    yield 'Random forest (F1 score={:.3f})'.format(score), precision, recall
    # Generate the ROC curve for random forest
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)
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

    # testing the Ada boost classifier (another type of classifier, similar to AUC)
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
   
    model = classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test), pos_label=1)
    # P-R curve
    y_prob = classifier.decision_function(X_test)
    y_pred = model.predict(X_test)
    #final= train_data
   # for i in final:
    #    if i!= ID_name:
   #         final.drop([i], axis=1, inplace= True)
    #final['Predictions']= y_pred
    #print("Final")
    #print(final)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
    # generating the ROC curve for Ada metric
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_prob, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    print('Ada boost AUC: %.3f' % auc)
    
    # including the score in the title
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

    # plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ')

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    #improving layout with matplotlib
    plt.tight_layout()


    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()

# =====================================================================
if __name__ == '__main__':

    #connecting to files
    train_data = pd.read_excel(r"C:\Users\Pigi\Desktop\Erasmus Semester Lessons\Data Mining\Project Assignment -AXA\Data_DSC2019_STUDENTS\DSC2019_Training.xlsx",  sheet_name='Sheet1')
    test_data= pd.read_excel(r"C:\Users\Pigi\Desktop\Erasmus Semester Lessons\Data Mining\Project Assignment -AXA\Data_DSC2019_STUDENTS\DSC2019_Test.xlsx",  sheet_name='Sheet1')
            
    target_name= train_data.columns[-1]
    #encoding the test and train set
    train_data= encoding(train_data)
    test_data= encoding(test_data)
    
   
    train_data[target_name]= np.where(train_data[target_name]== 'Y', 1, 0)
    print("target column: '{i}'".format(i=train_data[target_name]))

    
    #print(train_data)
   
    #separating the train and the test data
    Y= train_data[target_name].values
    X= train_data.drop(labels=target_name,axis=1)
    print(X.shape)
    print(Y.shape)
    print(X)
      
    #I tried to make a validation set but the AUC went down, I will try again
    '''X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)'''
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=101)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    print(X_train.shape)
    print(y_train.shape)
    print(X_train)
    
    print(X_test.shape)
    print(y_test.shape)
    print(X_test)

    # Evaluate my_trainultiple classifiers on the data
    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))
    # Display the results
    print("Plotting the results")
    plot(results)
    
    #transferring to excel sheet 
    #I WILL FINISH WITH THIS TODAY