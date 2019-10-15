from buildData import buildData
from main import mycrossval

print(" Running q3i file..")

def q3i():
    
    dfBoston50, dfBoston75, dfDigits = buildData()
    x50 = dfBoston50.iloc[:, 0:13]
    y50 = dfBoston50.iloc[:, 14] # Take the target50 column as y and not the actualresponse column
    x75 = dfBoston75.iloc[:, 0:13]
    y75 = dfBoston75.iloc[:, 14] # Take the target 75 column as y and not the actualresponse column
    xDigits = dfDigits.iloc[:, 0:64]
    yDigits = dfDigits.iloc[:, 64]
    
    k =10
    method = "LinearSVC"
    
    print("Applying LinearSVC on Boston 50 Data : Q3 Part1")
    mycrossval(method, x50, y50, k)

    print("Applying LinearSVC on Boston75 Data : Q3 Part1")
    mycrossval(method, x75, y75, k)
    
    print("Applying LinearSVC on Digits Data : Q3 Part1")
    mycrossval(method, xDigits, yDigits, k)
    
    method = "SVC"
    print("Applying SVC on Boston 50 Data : Q3 Part1")
    mycrossval(method, x50, y50, k)

    print("Applying SVC on Boston75 Data : Q3 Part1")
    mycrossval(method, x75, y75, k)

    print("Applying SVC on Digits Data : Q3 Part1")
    mycrossval(method, xDigits, yDigits, k)
    
    method = "LogisticRegression"
    print("Applying LogisticRegression on Boston 50 Data : Q3 Part1")
    mycrossval(method, x50, y50, k)

    print("Applying LogisticRegression on Boston75 Data : Q3 Part1")
    mycrossval(method, x75, y75, k)

    print("Applying LogisticRegression on Digits Data : Q3 Part1")
    mycrossval(method, xDigits, yDigits, k)

print("Completed Running q3i file")