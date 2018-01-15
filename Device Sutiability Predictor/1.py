import pandas as pd
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler,SMOTE
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
data_file=pd.read_csv('class.csv')
trainx= data_file.iloc[:, 0:12]
trainy= data_file.iloc[:, 12]
ros = SMOTE()
trainx,trainy= ros.fit_sample(trainx, trainy)
X_train, X_test, Y_train, Y_test = train_test_split(trainx, trainy, train_size=0.80,test_size=0.20, random_state=42)
model=GradientBoostingClassifier()
model.fit(X_train, Y_train.ravel())
result = model.predict(X_test)
print(classification_report(Y_test, result))
