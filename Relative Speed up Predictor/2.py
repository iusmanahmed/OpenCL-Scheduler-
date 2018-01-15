import numpy 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import pandas

data_file=pandas.read_csv('speedup_exp.csv')
Input =numpy.array(data_file.iloc[1:140, 2:26])
Output= numpy.array(data_file.iloc[1:140, 26])

trainx=Input[0:120]
trainy=Output[0:120]
testx=Input[120:140]
testy=Output[120:140]
est = GradientBoostingRegressor().fit(trainx, trainy)
print "mean_squared_error",mean_squared_error(testy, est.predict(testx)) 