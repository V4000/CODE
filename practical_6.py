
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("iris.csv")

x=dataset.iloc[ : ,:4 ].values
y=dataset['species'].values
print(dataset.head(5))
print(x)
print(y)

from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)
print(x_test)


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
print(classifier)


y_pred=classifier.predict(x_test)
print(y_pred)

from sklearn.metrices import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.matrices import accuracy_score
print("Accuracy : ",accuracy_score(y_test,y_pred))
print(cm)


df=pd.DataFrame({'Real Value'})