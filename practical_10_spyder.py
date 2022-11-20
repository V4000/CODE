import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Iris.csv")
print(df.tail())
print(df.dtypes)
sns.catplot(x ="variety", hue ="sepal.length",
kind ="count", data = df)

sns.histplot(data=df)