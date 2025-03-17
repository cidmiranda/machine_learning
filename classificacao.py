import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier

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
y_credit = base_credit.iloc[:, 4].values
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
#grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
#grafico.show()
X_census = base_census.iloc[:, 0:14].values
#print(X_census)
y_census = base_census.iloc[:, 14].values
#print(y_census)
#label_encoder_teste = LabelEncoder()
#print(X_census[:,1])
#teste = label_encoder_teste.fit_transform(X_census[:,1])
#print(teste)
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])
#print(X_census)
#print(len(np.unique(base_census['workclass'])))
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0,1,3,5,6,7,8,9,13])], remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census).toarray()
#print(X_census)
#print(X_census.shape)
#print(X_census[0])
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
#print(X_census[0])
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)
#print(X_credit_treinamento.shape)

X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census, y_census, test_size=0.15, random_state=0)
#print(X_census_treinamento.shape, y_census_treinamento.shape)
#print(X_census_teste.shape, y_census_teste.shape)
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)
with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)
base_risco_credito = pd.read_csv('risco_credito.csv')
#print(base_risco_credito)
X_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()
X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantia.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])
with open('risco_credito.pkl', mode='wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)
previsao = naive_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
#print(previsao)
#print(naive_risco_credito.classes_)
#print(naive_risco_credito.class_count_)
#print(naive_risco_credito.class_prior_)
with open('credit.pkl', mode='rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
#print(X_credit_treinamento.shape, y_credit_treinamento.shape)
#print(X_credit_teste.shape, y_credit_teste.shape)
naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)
#previsoes = naive_credit_data.predict(X_credit_teste)
#print(previsoes)
#print(y_credit_teste)
#print(accuracy_score(y_credit_teste, previsoes))
#print(confusion_matrix(y_credit_teste, previsoes))
cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
#plt.show()
#print(classification_report(y_credit_teste, previsoes))
with open('census.pkl', mode='rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)
#print(X_census_treinamento.shape, y_census_treinamento.shape)
#print(X_census_teste.shape, y_census_teste.shape)
naive_census_data = GaussianNB()
naive_census_data.fit(X_census_treinamento, y_census_treinamento)
previsoes = naive_census_data.predict(X_census_teste)
#print(previsoes)
#print(y_census_teste)
#print(accuracy_score(y_census_teste, previsoes))
cm = ConfusionMatrix(naive_census_data)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
#plt.show()
#print(classification_report(y_census_teste, previsoes))
with open('risco_credito.pkl', mode='rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)
#print(X_risco_credito)
#print(y_risco_credito)
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(X_risco_credito, y_risco_credito)
previsores = ['história','dívida','garantias','renda']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=[10,10])
tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_, filled=True)
#plt.show()
previsoes = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes)