import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

base_credit = pd.read_csv('credit_data.csv')

#print(base_credit)
#print(base_credit.head(10))
#print(base_credit.tail(5))
#print(base_credit.describe())
#print(base_credit[base_credit['loan'] >= 13000.000])
#print(base_credit[base_credit['loan'] <= 1.377630])
#print(np.unique(base_credit['default'], return_counts=True))
#sns.countplot(x = base_credit['default'])
#plt.show()
#plt.hist(x = base_credit['age'])
#plt.show()
#plt.hist(x = base_credit['income'])
#plt.show()
#plt.hist(x = base_credit['loan'])
#plt.show()
#grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
#grafico.show()
#print(base_credit.loc[base_credit['age'] < 0])
#base_credit2 = base_credit.drop('age', axis = 1)
#print(base_credit2)
#base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
#print(base_credit3)
#print(base_credit.mean())
#print(base_credit['age'].mean())
#print(base_credit['age'][base_credit['age'] > 0].mean())
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92
#print(base_credit.head(27))
#print(base_credit.loc[base_credit['age'] == 40.92])
#plt.hist(x = base_credit['age'])
#plt.show()
#grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
#grafico.show()
#print(base_credit.isnull())
#print(base_credit.isnull().sum())
#print(base_credit.loc[pd.isnull(base_credit['age'])])
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
#print(base_credit.loc[pd.isnull(base_credit['age'])])
#print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])
X_credit = base_credit.iloc[:, 1:4].values
#print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
#y_credit = base_credit.iloc[:, 4].values
#print(y_credit)
#print(X_credit[:, 0].min())
#print(X_credit[:, 0].max())
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
#print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
base_census = pd.read_csv('census.csv')
#print(base_census)
#print(base_census.describe())
#print(base_census.isnull().sum())
#print(np.unique(base_census['income'], return_counts=True))
#sns.countplot(x = base_census['income'])
#plt.show()
#grafico = px.treemap(base_census, path=['workclass', 'age'])
#grafico.show()
grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
grafico.show()