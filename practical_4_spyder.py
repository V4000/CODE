import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
boston_dataset=load_boston()
print(boston_dataset.keys())
data=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
print(data.head())
data['price']=boston_dataset.target
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
sns.displot(data['price'],bins=30)
plt.show