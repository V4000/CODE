import pandas as pd
iris = pd.read_csv('/home/cg/Desktop/iris.data',header=None)
col_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']
iris = pd.read_csv('/home/cg/Desktop/Yash 3254/iris.data',names=col_names )

print(iris.head(n=5))

print(iris.tail(n=5))

print(iris.columns)

print(iris.index)

print(iris.shape)

print(iris.dtypes)

print(iris.columns.values)

print(iris.describe(include='all'))

print(iris[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']])

print(iris.sort_values(by=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']))
print(iris.iloc[6])
print(iris[0:3])
print(iris.iloc[1:4])
print(iris.iloc[3:5,0:2])
print(iris.iloc[[1, 2,4], [0, 2]])
print(iris.iloc[1:3, :])
print(iris.iloc[:, 1:3])
