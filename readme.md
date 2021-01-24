# Binary & Multi class classification
## Datasets 
The 2-class classification datasets from Boston50, Boston75, extracted from [Boston Housing dataset](https://github.com/rupakc/UCI-Data-Analysis/tree/master/Boston Housing Dataset/Boston Housing) and the 10-class classification [dataset](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) from Digits

## Objectives
- Develop code for __k-fold cross validation__ without using sci-kit learn. Report the error rates in each fold as well as the mean and standard devia-
tion of error rates across folds for the three methods: LinearSVC, SVC, and LogisticRegression, applied to the three classification datasets: Boston50, Boston75, and Digits.
- Develop code for __train-test split__, which performs random splits on the data (X, y) so that π ∈ [0, 1] fraction of the data is used for training using method, rest is used for testing, and the process is repeated k times, after which the code returns the error rate for each such train-test split. Using this function, report the mean and standard deviation of error rate for the three methods: LinearSVC, SVC, and LogisticRegression, applied to the three classification
- Perform __Feature Engineering__ on Digits Dataset and report results for LinearSVC, SVC, and LogisticRegression

## Additional Information

- main.py file: Has code for classifiers
- buildData.py file: builds boston50, boston75 and digits data sets
- q3i.py file: Has 9 method calls for q 3 part 1
- q3ii.py file: Has 9 method calls for q 3 part 2
- q4.py file: Has 6 method calls for q 4
