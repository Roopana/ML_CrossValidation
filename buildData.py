from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

print(" Running buildData file..")
def buildData():
    boston = datasets.load_boston()
    ## Create Boston 50 data set
    ## responseActual column corresponds to the actual response values
    ##response50 column corresponds to the values as per 50th percentile
    dfBoston50 = pd.DataFrame(data=boston['data'], columns = boston['feature_names'])
    dfBoston50['responseActual'] = boston['target']
    dfBoston50['response50'] = dfBoston50['responseActual']>= np.percentile(dfBoston50['responseActual'],50)
    dfBoston50['response50'] = dfBoston50['response50']*1
    ##Create Boston75 data set
    ## responseActual column corresponds to the actual response values
    ##response75 column corresponds to the values as per 75th percentile
    dfBoston75 = pd.DataFrame(data=boston['data'], columns = boston['feature_names'])
    dfBoston75['responseActual'] = boston['target']
    dfBoston75['response75'] = dfBoston75['responseActual']>= np.percentile(dfBoston50['responseActual'],75)
    dfBoston75['response75'] = dfBoston75['response75']*1
    #Construct dfDigits Data
    digits = load_digits(n_class=10, return_X_y=False)
    dfDigits = pd.DataFrame(data=digits.data)
    dfDigits['response'] = digits.target
    
    return dfBoston50, dfBoston75, dfDigits

print(" Completed running buildData file..")