import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()