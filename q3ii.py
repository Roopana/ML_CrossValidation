
from buildData import buildData
from main import my_train_test

print(" Running q3ii file..")
def q3ii():
    dfBoston50, dfBoston75, dfDigits = buildData()
    #Boston 50 Data Set
    x50 = dfBoston50.iloc[:, 0:13]
    y50 = dfBoston50.iloc[:, 14] 
    # Boston 75 Dat Set
    x75 = dfBoston75.iloc[:, 0:13]
    y75 = dfBoston75.iloc[:, 14]
    # Digits Data Set
    xDigits = dfDigits.iloc[:, 0:64]
    yDigits = dfDigits.iloc[:, 64]
    
    k =10
    pi = 0.75
    method = "LinearSVC"
    
    # Running Linear SVC on Boston 50, Boston 75 and digits data sets
    print("Applying LinearSVC on Boston50 Data - Q3 Part2")
    my_train_test(method, x50, y50, pi, k)

    print("Applying LinearSVC on Boston75 Data - Q3 Part2")
    my_train_test(method, x75, y75, pi, k)

    print("Applying LinearSVC on Digits Data - Q3 Part2")
    my_train_test(method, xDigits, yDigits, pi, k)
   
    # Running SVC on Boston 50, Boston 75 and digits data sets
    method = "SVC"
    print("Applying SVC on Boston50 Data - Q3 Part2")
    my_train_test(method, x50, y50, pi, k)

    print("Applying SVC on Boston75 Data - Q3 Part2")
    my_train_test(method, x75, y75, pi, k)

    print("Applying SVC on Digits Data - Q3 Part2")
    my_train_test(method, xDigits, yDigits, pi, k)
    
    # Running Logistic Regression Classifier on Boston 50, Boston 75 and digits data sets
    method = "LogisticRegression"
    print("Applying Logistic Regression on Boston50 Data - Q3 Part2")
    my_train_test(method, x50, y50, pi, k)

    print("Applying Logistic Regression on Boston75 Data - Q3 Part2")
    my_train_test(method, x75, y75, pi, k)

    print("Applying Logistic Regression on Digits Data - Q3 Part2")
    my_train_test(method, xDigits, yDigits, pi, k)
    
print("Completed running q3ii file")