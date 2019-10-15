import numpy as np
import pandas as pd
from buildData import buildData
from main import rand_proj, quadproj, mycrossval

print(" Running q4 file..")
dfBoston50, dfBoston75, dfDigits = buildData()

def q4():
    d =32 # the value is given
    print("Applying Feature Engineering on Digits Data")
    x = dfDigits.iloc[:, 0:64]
    y = dfDigits.iloc[:, 64]
    G = rand_proj(x,d)
    x1 = np.dot(x.values,G)
    x2= quadproj(x)
    k =10 # Given k= 10
    
    print("Applying Linear SVC on Digits Data with X1: Q4")
    mycrossval("LinearSVC", pd.DataFrame(x1), y, k)

    print("Applying Linear SVC on Digits Data with X2: Q4")
    mycrossval("LinearSVC", pd.DataFrame(x2), y, k) 
    
    print("Applying SVC on Digits Data with X1: Q4")
    mycrossval("SVC", pd.DataFrame(x1), y, k)
    
    print("Applying SVC on Digits Data with X2: Q4")
    mycrossval("SVC", pd.DataFrame(x2), y, k)

    print("Applying LogisticRegression on Digits Data with X1: Q4")
    mycrossval("LogisticRegression", pd.DataFrame(x1), y, k)

    print("Applying LogisticRegression on Digits Data with X2: Q4")
    mycrossval("LogisticRegression", pd.DataFrame(x2), y, k)

   
print("Completed Running q4 file")