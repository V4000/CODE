import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Social_Network_Ads.csv")
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
print(df.info())
sns.countplot(df['Purchased'])
plt.title('Distribution of Purchased or not')
plt.xlabel('Purchased or not')
plt.ylabel('Frequency')
# plt.show()

plt.figure(figsize = (10,6))
plt.hist(df['Age'], bins  = 6, color = 'blue', rwidth = 0.98)
plt.title('Distribution of Age')
plt.xlabel('Different Ages')
plt.ylabel('Frequency')
# plt.show()
X = df.iloc[:,[2,3]].values
print(X)

y = df.iloc[:,4].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
#test_size =0.25 means 25% data of whole dataset will be used for training and rest of for testing

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_test,y_pred)
print('Confusion Matrix: ')
print(cm1)   
ac1 = accuracy_score(y_test, y_pred)*100
print('Accuracy Score:')
print(ac1)

tp=cm1[0][0]
tn=cm1[1][1]
fp=cm1[1][0]
fn=cm1[0][1]
total=tp+tn+fp+fn

error_rate=(fp+fn)/(total)
print('error rate: ')
print(error_rate)

from sklearn.metrics import classification_report
print('                        classification report:')
print('')
print(classification_report(y_test,y_pred))