Contains implementation for K cross Validaion
Compares performance of Logistic Regression, SVC, Linear SVC on 2 data sets (Multiclass- Digits dataset and binary class - Boston50 & Boston70)

*********************************************************************************************************************

Steps to be followed to run

*********************************************************************************************************************
# step 1: Change directory to the location where .py files are located:
import os
os.chdir("/home/vuppa007/Downloads")

#Step 2: Run run.py file. It calls the wrapper methods to evaluate q3 & q4.
exec(open('run.py').read())

********************************************************************************************************************

Additional Information about wrapper calls and Main file:

main.py file: Has code for classifiers

buildData.py file: builds boston50, boston75 and digits data sets

q3i.py file: Has 9 method calls for q 3 part 1

q3ii.py file: Has 9 method calls for q 3 part 2

q4.py file: Has 6 method calls for q 4


## Datasets 
The 2-class classification datasets from Boston50, Boston75, extracted from [Boston Housing dataset](https://github.com/rupakc/UCI-Data-Analysis/tree/master/Boston Housing Dataset/Boston Housing) and the 10-class classification [dataset](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) from Digits

## Objectives
- Develop code for k-fold cross validation without using sci-kit learn. Report the error rates in each fold as well as the mean and standard devia-
tion of error rates across folds for the three methods: LinearSVC, SVC, and LogisticRegression, applied to the three classification datasets: Boston50, Boston75, and Digits.
- Develop code for my train test(method,X,y,π,k), which performs random splits on the data (X, y) so that π ∈ [0, 1] fraction of the data is used for training using method, rest is used for testing, and the process is repeated k times, after which the code returns the error rate for each such train-test split. Using my train test, with π = 0.75 and k = 10, report the mean and standard deviation of error rate for the three methods: LinearSVC, SVC, and LogisticRegression, applied to the three classification
- Feature Engineering on Digits Dataset and report results for LinearSVC, SVC, and LogisticRegression

