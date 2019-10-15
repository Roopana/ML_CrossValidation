import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

print("Running Main file..")

#Given an array X and the K folds this function gives the training and test sets

def train_test_split_custom(x, k, ithSplit):
    
    indexSplit = int(x.shape[0]/k) #gives floor of the floating point number x.shape[0]/10
    startIndex_testData = ithSplit*indexSplit; # calculate start Index of test split
    len = x.shape[0]
    endIndex_testData = min(startIndex_testData + indexSplit, len) # calculate end index of test split

    test_data = x.iloc[startIndex_testData:endIndex_testData]

    train_data1 = x.iloc[0:startIndex_testData]
    train_data2 = x.iloc[endIndex_testData:len]
    
    #train data is formed by concatnating above two arrays
    train_data= pd.concat([train_data1, train_data2])

    return train_data, test_data

#Given list of predicted values, list of test values, returns the output as list of values that are misclassified
def getMisClassifiedResponses(y_test, y_pred):
    #Checks if two list are same
    correctlyClassifiedBooleanResponses = y_test == y_pred
    correctlyClassifiedResponses = correctlyClassifiedBooleanResponses*1
    return np.subtract(1, correctlyClassifiedResponses)

#Returns the error rate of test data as per input classifier - method
def classify(method, x_train, x_test, y_train, y_test):
    error = 0
    if(method == 'LogisticRegression'):
        logReg = LogisticRegression() 
        logReg.fit(x_train, y_train)
        y_pred = logReg.predict(x_test)
        #acc = logReg.score(x_test, y_test)
        misClassifiedResponses = getMisClassifiedResponses(y_test, y_pred)
        error = np.mean(misClassifiedResponses)
            
    elif(method =='LinearSVC'):
        linearSVC = LinearSVC()
        linearSVC.fit(x_train, y_train)  
        y_pred = linearSVC.predict(x_test)
        #acc= linearSVC.score(x_test, y_test)
        misClassifiedResponses = getMisClassifiedResponses(y_test, y_pred)
        error = np.mean(misClassifiedResponses)
        
    elif(method =='SVC'):
        sVC = SVC()
        sVC.fit(x_train, y_train)
        y_pred = sVC.predict(x_test)
        misClassifiedResponses = getMisClassifiedResponses(y_test, y_pred)
        #acc= sVC.score(x_test, y_test)
        error = np.mean(misClassifiedResponses)

    else:
        print('Invalid classifier Type. Only LogisticRegression, LinearSVC and SVC are valid types')
    
    return error

# Inputs : k - No of folds in Cross Validation, X - Input data and Y - response
# Returns and prints the error of the classifier 'method' for X,Y using Cross Validation
def mycrossval(method,x,y,k):
    error = []

    for ithSplit in range(k):
        # create training and testing vars
        x_train, x_test = train_test_split_custom(x, k, ithSplit)
        y_train, y_test = train_test_split_custom(y, k, ithSplit)
        error_i = classify(method, x_train, x_test, y_train, y_test)
        print("Error in Fold ",ithSplit, ": {:.2f}".format(error_i ))
        error.append(error_i)
                           
    print("Mean error: {:.2f}".format(np.mean(error)))
    print("Standard deviation of error : {:.2f}".format(np.std(error)))
       
    return error
 
# implementing test train split using ratio -pi

def train_test_split_ratio(x, y, pi):
    num_training_instances = int(pi * x.shape[0])
    #calculate random numbers from 0 to x.shape[0]
    randomIndices = np.random.permutation(x.shape[0])
    
    permutedX = np.take(x,randomIndices,axis=0)
    permutedY = np.take(y,randomIndices,axis=0)
    
    x_train= permutedX.iloc[0:num_training_instances]
    x_test = permutedX.iloc[num_training_instances:x.shape[0]]
    
    y_train= permutedY.iloc[0:num_training_instances]
    y_test = permutedY.iloc[num_training_instances:x.shape[0]]

    return x_train, x_test, y_train, y_test
    
# Inputs : K - No of folds in Cross Validation, X - Input data and Y - response, pi -  ratio of train data / test data split
# Calculates and prints the error of the classifier 'method' for X,Y using Cross Validation
def my_train_test(method,x,y,pi,k):
    error = []
    for ithSplit in range(k):
        # create training and testing vars
        x_train, x_test, y_train, y_test = train_test_split_ratio(x,y, pi)
        error_i = classify(method, x_train, x_test, y_train, y_test)
        print("Error in Fold ",ithSplit, ": {:.2f}".format( error_i ))
        error.append(error_i)
                           
    print("Mean error : {:.2f}".format(np.mean(error)))
    print("Standard deviation of error : {:.2f}".format(np.std(error)))
    return error   

#Question 4, part 1 Feature Engineering
# return random matrix of dimension:columns of x, d with elements from a uniform distribution over [0, 1).
def rand_proj(x,d):
    colsOfX = x.shape[1]
    #Create a matrix with rows =colsOfX and Cols= d  
    #populate it with random samples from a uniform distribution over [0, 1).
    randomMatrix = np.random.rand(colsOfX,d) 
    return randomMatrix


#Question 4 Part 2, Apply transformation 
# compute product of two columns X[i] & X[j] where i<j
def quadproj(x):
    num_cols = x.shape[1]
    count = num_cols;
    for i in range(0, num_cols):
        for j in range(i, num_cols):
            x[count] = x[i]*x[j]
            count = count+1
    return x

print("Completed Running Main file")