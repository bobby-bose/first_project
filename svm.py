Testing---------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# reading csv file and extracting class column to y.
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data=pd.read_csv("Data4.csv",encoding='latin-1',low_memory=False)
columns=[0,1,2,3,7,8,9,10,27,28,29,26]
data=data[data.columns[columns]]
data=pd.DataFrame(data)
data.drop(['country_txt','region_txt','attacktype1_txt'],inplace=True,axis=1)
# print(data.isnull().sum())
X=data.iloc[:,:7].values
y  = data.iloc[:,8].values # classes having 0 and 1
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=4, random_state=4)
# extracting two features
clf = SVC(kernel='linear')
# fitting x samples and y classes
clf.fit(x_train, y_train)
# 569 samples and 2 features
print("Completed First Step")
print(clf.predict(x_test))
print(accuracy_score(y_test,clf.predict(x_test)))
