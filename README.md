# Machine Learning

## Métodos preditivos

### Classificação (rótulos)

#### Marketing direto
#### Insatisfação de clientes
#### Risco de crédito
#### Filtros de spam
#### Separação de notícias
#### Reconhecimento de voz
#### Reconhecimento de face
#### Previsão de doenças

### Regressão (números)

#### Gastos propaganda -> valor de venda
#### Temperatura, umidade e pressão do ar -> velocidade do vento
#### Fatores externos -> valor do dólar
#### Resultados do exame -> probabilidade de um paciente sobreviver
#### Risco de investimento
#### Gastos no cartão de crédito, histórico -> limite
#### Valores anteriores -> valores de produtos

## Métodos Descritivos

### Associação

#### Prateleiras de mercado
#### Promoções com itens que são vendidos em conjunto
#### Planejar catálogos de lojas e folhetos de promoções
#### Controle de evasão em universidades

### Agrupamento

#### Segmentação de mercado
#### Agrupamento de documentos / notícias
#### Agrupamento de produtos similares
#### Perfil de clientes (netflix)
#### Análise de redes sociais

### Detecção de desvios (outliers)

#### Fraude em cartão de crédito
#### Intrusão em redes
#### Uso de energia elétrica, água ou telefone
#### Desempenho de atletas (doping)
#### Monitorar máquinas em um data center

### Padrões sequenciais

#### Livrarias, lojas de equipamentos de atletismo, computadores
#### Marketing para adquirir um novo produto
#### Previsão de doenças
#### Navegação em sites

### Sumarização

#### São ouvintes do programa, homens na faixa de 25 a 35 anos, com nível superior e que trabalham na área de administração
#### Segmentação de mercado

## Tipos de apredizagem de máquina

### Supervisionada

#### Supervisor ajuda o algoritmo a aprender

### Não supervisionada

#### Analisar automaticamente os dados (associação, agrupamento)
#### Necessita análise para determinar o significado dos padrões encontradas

### Reforço

#### Aprender com as interações com o ambiente (causa e efeito)

## Classificação - Pré-processamento com pandas e scikit-learn

### Variáveis

##### Numéricas
###### Contínua (números reais. temperatura, altura, peso, salário)
###### Discreta (conjuto de valores finitos (inteiros)) Contagem
##### Categóricas
###### Nominal (dados não mensuráveis, sem ordenação: cor dos olhos, gênero, nome)
###### Ordinal (Categorização sob uma ordenação. Tamanho P, M, G)

### Base de crédito

```bash
pip -q install plotly
pip -q install yellowbrick
```
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```
Carregar a base de dados
```bash
base_credit = pd.read_csv('credit_data.csv')
```
Visualizar os 5 primeiros registros
```bash
print(base_credit.head(5))
```
| clientid |    income    |       age | loan        | default |
|:---------|:------------:|----------:|-------------|--------:|
| 1        | 66155.925095 | 59.017015 | 8106.532131 |       0 |
| 2        | 34415.153966 | 48.117153 | 6564.745018 |       0 |
| 3        | 57317.170063 | 63.108049 | 8020.953296 |       0 |
| 4        | 42709.534201 | 45.751972 | 6103.642260 |       0 |
| 5        | 66952.688845 | 18.584336 | 8770.099235 |       1 |

Visualizar os 5 últimos registros
```bash
print(base_credit.tail(5))
```
| clientid |    income    |       age | loan        | default |
|:---------|:------------:|----------:|-------------|--------:|
| 1996     | 59221.044874 | 48.518179 | 1926.729397 |       0 |
| 1996     | 69516.127573 | 23.162104 | 3503.176156 |       0 |
| 1998     | 44311.449262 | 28.017167 | 5522.786693 |       1 |
| 1999     | 43756.056605 | 63.971796 | 1622.722598 |       0 |
| 2000     | 69436.579552 | 56.152617 | 7378.833599 |       0 |

Estatísticas
```bash
print(base_credit.describe())
```
|       |    clientid |       income |         age |         loan |     default |
|-------|------------:|-------------:|-------------|-------------:|------------:|
| count | 2000.000000 |  2000.000000 | 1997.000000 |  2000.000000 | 2000.000000 |
| mean  | 1000.500000 | 45331.600018 |   40.807559 |  4444.369695 |    0.141500 |
| std   |  577.494589 | 14326.327119 |   13.624469 |  3045.410024 |    0.348624 |
| min   |    1.000000 | 20014.489470 |  -52.423280 |     1.377630 |    0.000000 |
| 25%   |  500.750000 | 32796.459717 |   28.990415 |  1939.708847 |    0.000000 |
| 50%   | 1000.500000 | 45789.117313 |   41.317159 |  3974.719419 |    0.000000 |
| 75%   | 1500.250000 | 57791.281668 |   52.587040 |  6432.410625 |    0.000000 |
| max   | 2000.000000 | 69995.685578 |   63.971796 | 13766.051239 |    1.000000 |

Filtro de quem deve mais de 10.000
```bash
print(base_credit[base_credit['loan'] >= 13000.000])
```
| clientid |    income    |       age | loan         | default |
|:---------|:------------:|----------:|--------------|--------:|
| 481      | 66049.934032 | 29.315767 | 13172.681298 |       1 |
| 768      | 67520.759597 | 45.415624 | 13041.779452 |       0 |
| 1051     | 69456.567771 | 48.053557 | 13190.365886 |       0 |
| 1351     | 69592.010828 | 63.238625 | 13025.056571 |       0 |
| 1379     | 69755.320163 | 44.543682 | 13766.051239 |       0 |

Filtro de quem deve menos de 1.377630
```bash
print(base_credit[base_credit['loan'] <= 1.377630])
```
| clientid |    income    |       age | loan    | default |
|:---------|:------------:|----------:|---------|--------:|
| 866      | 28072.604355 | 54.142548 | 1.37763 |       0 |

### Visualização dos dados
Contagem de quantos registros existem em cada uma das classes.
Classe 0, pagou o empréstimo e classe 1, não pagou o empréstimo.
```bash
print(np.unique(base_credit['default'], return_counts=True))
```
(array([0, 1], dtype=int64), array([1717,  283], dtype=int64))

Gerar gráfico de contagem de registros das classes
```bash
sns.countplot(x = base_credit['default'])
plt.show()
```
![Alt text](imgs/classes.png "Classe 0, pagou o empréstimo e classe 1, não pagou o empréstimo")

Histograma de idade
```bash
plt.hist(x = base_credit['age'])
plt.show()
```
![Alt text](imgs/hist_age.png "Histograma de idade")

Histograma de renda
```bash
plt.hist(x = base_credit['income'])
plt.show()
```
![Alt text](imgs/hist_income.png "Histograma de renda")

Histograma de dívida
```bash
plt.hist(x = base_credit['loan'])
plt.show()
```
![Alt text](imgs/hist_divida.png "Histograma de dívida")

Gráfico de dispersão
```bash
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()
```
![Alt text](imgs/dispersao.png "Dispersão")

### Tratamento de valores inconsistentes
verificar clientes com idade menor que zero
```bash
print(base_credit.loc[base_credit['age'] < 0])
```
| clientid |    income    |        age | loan        | default |
|:---------|:------------:|-----------:|-------------|--------:|
| 16       | 50501.726689 | -28.218361 | 3977.287432 |       0 |    
| 22       | 32197.620701 | -52.423280 | 4244.057136 |       0 |
| 27       | 63287.038908 | -36.496976 | 9595.286289 |       0 |

Podemos apagar a coluna inteira

Vamos criar uma nova variável pra receber a nova base
```bash
base_credit2 = base_credit.drop('age', axis = 1)
print(base_credit2)
```
| clientid |    income    | loan        | default |
|:---------|:------------:|-------------|--------:|
| 1        | 66155.925095 | 8106.532131 |       0 |
| 2        | 34415.153966 | 6564.745018 |       0 |
| 3        | 57317.170063 | 8020.953296 |       0 |
| 4        | 42709.534201 | 6103.642260 |       0 |
| 5        | 66952.688845 | 8770.099235 |       1 |
...

Podemos apagar os registros com valores inconsistentes

Vamos criar uma nova variável pra receber a nova base
```bash
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
print(base_credit3)
```
| clientid |    income    |       age | loan        | default |
|:---------|:------------:|----------:|-------------|--------:|
| 1        | 66155.925095 | 59.017015 | 8106.532131 |       0 |
| 2        | 34415.153966 | 48.117153 | 6564.745018 |       0 |
| 3        | 57317.170063 | 63.108049 | 8020.953296 |       0 |
| 4        | 42709.534201 | 45.751972 | 6103.642260 |       0 |
| 5        | 66952.688845 | 18.584336 | 8770.099235 |       1 |
...

Podemos preencher os valores inconsistentes manualmente com as médias de idade

```bash
print(base_credit.mean())
```
clientid     1000.500000
income      45331.600018
age            40.807559
loan         4444.369695
default         0.141500
dtype: float64

```bash
print(base_credit['age'].mean())
```
40.80755937840458

Está calculando com as idades negativas, vamos corrigir
```bash
print(base_credit['age'][base_credit['age'] > 0].mean())
```
40.92770044906149

Vamos atualizar os valores
```bash
print(base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92 )
```
Realizar uma nova consulta
```bash
print(base_credit.loc[base_credit['age'] < 0])
```
Empty DataFrame
Columns: [clientid, income, age, loan, default]
Index: []

```bash
base_credit.loc[base_credit['age'] == 40.92]
```
| clientid |    income    |   age | loan        | default |
|:---------|:------------:|------:|-------------|--------:|
| 16       | 50501.726689 | 40.92 | 3977.287432 |       0 |    
| 22       | 32197.620701 | 40.92 | 4244.057136 |       0 |
| 27       | 63287.038908 | 40.92 | 9595.286289 |       0 |

Histograma de idade
```bash
plt.hist(x = base_credit['age'])
plt.show()
```
![Alt text](imgs/hist_age2.png "Histograma de idade")

Gráfico de dispersão
```bash
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()
```
![Alt text](imgs/dispersao2.png "Dispersão")

### Tratamento de valores faltantes
```bash
print(base_credit.isnull())
```
![Alt text](imgs/isNull.png "Valores nulos")

```bash
print(base_credit.isnull().sum())
```
![Alt text](imgs/isNull2.png "Valores nulos")

```bash
print(base_credit.loc[pd.isnull(base_credit['age'])])
```
![Alt text](imgs/isNull3.png "Valores nulos")

```bash
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
print(base_credit.loc[pd.isnull(base_credit['age'])])
```
Empty DataFrame
Columns: [clientid, income, age, loan, default]
Index: []

```bash
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])
```
![Alt text](imgs/isin.png "Valores nulos")

### Divisão entre previsores e classe
Vamos usar iloc para pegar os valores de todas as linhas ":" da coluna 1 (income) até a coluna 3 (loan) "1:4"
Previsores
```bash
X_credit = base_credit.iloc[:, 1:4].values
print(X_credit)
```
![Alt text](imgs/iloc.png "iloc")

Vamos pegar os valoes da classe, coluna 4 (default)
Classes
```bash
y_credit = base_credit.iloc[:, 4].values
print(y_credit)
```
![Alt text](imgs/iloc2.png "iloc")

### Escalonamento dos atributos
Evitar que o algoritmo interprete a importância dos atributos errado
Pessoa com a menor renda
```bash
print(X_credit[:, 0].min())
```
![Alt text](imgs/min.png "min")
Para as outras colunas, é só mudar o indice
```bash
print(X_credit[:, 1].min()) #idade
print(X_credit[:, 2].min()) #divida
```

Pessoa com a maior renda
```bash
print(X_credit[:, 0].max())
```
![Alt text](imgs/max.png "max")
Para as outras colunas, é só mudar o indice
```bash
print(X_credit[:, 1].max()) #idade
print(X_credit[:, 2].max()) #divida
```
Deixar os valores na mesma escala

Padronização (Standardisation)
```bash
x = x - media(x) / desvio padrão(x)
```
Normalização (Normalization)
```bash
x = x - minimo(x) / maximo(x) - minimo(x)
```
Usando padronização
```bash
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
```
![Alt text](imgs/scaler_after.png "deis")
Antes da transformação
![Alt text](imgs/scaler_before.png "antes")

### Base de dados do censo
Outras bases: http://archive.ics.uci.edu/ml/datasets/adult

```bash
base_census = pd.read_csv('census.csv')
print(base_census)
```
![Alt text](imgs/census.png "censo")

Estatísticas
```bash
print(base_census.describe())
```
![Alt text](imgs/desc_census.png "censo")

Existe valores faltantes?
```bash
print(base_census.isnull().sum())
```
![Alt text](imgs/null_census.png "censo")

Não precisamos ajustar os dados da base de dados

### Visualização dos dados
Visualizar os dados da nossa classe
```bash
print(np.unique(base_census['income'], return_counts=True))
```
![Alt text](imgs/classe_count.png "Visualização")
```bash
sns.countplot(x = base_census['income'])
plt.show()
```
![Alt text](imgs/plot_income.png "Visualização")
```bash
grafico = px.treemap(base_census, path=['workclass'])
grafico.show()
```
![Alt text](imgs/plot_workclass.png "Visualização")
```bash
grafico = px.treemap(base_census, path=['workclass', 'age'])
grafico.show()
```
![Alt text](imgs/plot_workclass2.png "Visualização")
```bash
grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
grafico.show()
```
![Alt text](imgs/plot_relation.png "Visualização")

### Divisão entre previsores e classes

Vamos usar iloc para pegar os valores de todas as linhas ":" da coluna 0 (age) até a coluna 13 (native-country) "1:14"
Previsores
```bash
X_census = base_census.iloc[:, 0:14].values
print(X_census)
```
![Alt text](imgs/x_census.png "iloc")

Vamos pegar os valoes da classe, coluna 14 (income)
Classes
```bash
y_census = base_census.iloc[:, 14].values
print(y_census)
```
![Alt text](imgs/y_census.png "iloc")

### Atributos categóricos - LabelEncoder (string para números)
```bash
label_encoder_teste = LabelEncoder()
print(X_census[:,1])
teste = label_encoder_teste.fit_transform(X_census[:,1])
print(teste)
```
Antes

![Alt text](imgs/categoric2.png "atributos categóricos")
Depois

![Alt text](imgs/categoric1.png "atributos categóricos")

Transformar cada atributo
```bash
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

print(X_census)
```
![Alt text](imgs/x_census2.png "atributos categóricos")

### Atributos categóricos - OneHotEncoder (evitar que o algoritmo interprete erroneamente o indice do array como peso no cálculo)
Exemplo
```bash
# Carro

#Gol  Pálio Uno
#  1      2   3

#Gol    1 0 0
#Pálio  0 1 0
#Uno    0 0 1 #
```
1 coluna por categoria
```bash
print(len(np.unique(base_census['workclass'])))

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0,1,3,5,6,7,8,9,13])], remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census).toarray()
print(X_census)
```
![Alt text](imgs/x_census3.png "atributos categóricos")
```bash
print(X_census.shape)
```
![Alt text](imgs/x_census4.png "atributos categóricos")

Agora temos 108 colunas na base

### Escalonamento dos atributos (evitar que o algoritmo interprete erroneamente o valor da variável muito mais alto do que outra variável, como peso no cálculo)
Antes
```bash
print(X_census[0])
```
![Alt text](imgs/escala1.png "escalonamento")
Depois
```bash
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
print(X_census[0])
```
![Alt text](imgs/escala2.png "escalonamento")

### Introdução a avaliação de algoritmos
O algorítmo avalia os dados e gera um modelo baseado nos dados
Divisão da base entre teinamento e teste. Não podem ter os mesmos dados para não enganar o algorítmo

Ex: Naive Bayes

Gera tabela de probabilidades baseado no treinamento. 
Submente os registros de teste ao algorítmo para comparar com o treinamento.
Baseado nos acertos, é calculada a taxa de acerto: número de acertos / número de registros
0.75 * 100 = 75%

### Base de treinamento e teste

Para realizar a divisão da base de dados, executamos o comando abaixo, o mais importante é o random_state, que garante que os dados serão gerados sempre iguais, os mesmo registros em treinamento e teste
O parâmetro test_size é o tamanho da base dados de teste, 25%, no comando abaixo (0.25)
Base de crédito
```bash
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)
print(X_credit_treinamento.shape)
```
![Alt text](imgs/div1.png "Divisão bases")
Base do censo
```bash
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census, y_census, test_size=0.15, random_state=0)
print(X_census_treinamento.shape, y_census_treinamento.shape)
print(X_census_teste.shape, y_census_teste.shape)
```
![Alt text](imgs/div2.png "Divisão bases")
A quantidade de registros na na base de teste e teinamento, precisam estar iguais (27676 = 27676, 4885 = 4885)

### Salvar as bases de dados
Vamos usar o pickle
```bash
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)

with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)
```
Vai criar dois aquivos, credit.pkl e census.pkl.

Para mais exemplos sobre pré-processamento de bases de dados, consulte: https://scikit-learn.org/stable/modules/preprocessing.html

E também:

Livro Python Machine Learning de Sebastian Raschka: o capítulo 4 do livro é sobre a construção de boas bases de dados utilizando técnicas de pré-processamento

Livro Machine Learning with Python de Chris Albon: os capítulos iniciais do livro apresentam uma variedade grande de técnicas de pré-processamento

Capítulo Data Quality de Tamraparni Dasu e Theodore Johnson da série de livros Wiley Series in Probability and Statistics: ótimo complemento depois da aprendizagem das técnicas de pré-processamento

## Aprendizagem bayesiana
Em teoria das probabilidades e estatística, o teorema de Bayes (alternativamente, a lei de Bayes ou a regra de Bayes) descreve a probabilidade de um evento acontecer, com base em um conhecimento que pode estar relacionado ao evento.

### Introdução
Baseado em probabilidade, principalmente texto. (Teorema de Bayes)
Exemplos:

Filtros de spam, mineração de emoções, separação de documentos

Ele gera uma tabela de probabilidades

Vantagens: Rápido, Simplicidade de interpretação, Trabalha com altas dimensões, Boas previsões em bases pequenas
Desvantagens: Combinação de características (atributos independentes) - cada par de características são independentes - nem sempre é verdade

### Base risco de crédito

```bash
base_risco_credito = pd.read_csv('risco_credito.csv')
print(base_risco_credito)
```
![Alt text](imgs/bay1.png "Bayes")

Divisão previsores e classe
```bash
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
```
Como a base é pequna não vamos aplicar o OneHotEncoder

Salvado a base
```bash
with open('risco_credito.pkl', mode='wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)
```
Criando o algorítimo
```bash
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

# história boa (0), dívida alta (0), garantias nenhuma (1), renda > 35 (2)
# história ruim (2), divida alta (0), garantias adequada (0), renda < 15 (0)

previsao = naive_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsao)
```
![Alt text](imgs/bay2.png "Bayes")

```bash
print(naive_risco_credito.classes_)
print(naive_risco_credito.class_count_)
print(naive_risco_credito.class_prior_)
```
![Alt text](imgs/bay3.png "Bayes")

### Base crédito
```bash
with open('credit.pkl', mode='rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
print(X_credit_treinamento.shape, y_credit_treinamento.shape)
print(X_credit_teste.shape, y_credit_teste.shape)
```
![Alt text](imgs/bay4.png "Bayes")

Treinando e executando o algorítmo
```bash
naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)
previsoes = naive_credit_data.predict(X_credit_teste)
print(previsoes)
```
![Alt text](imgs/bay5.png "Bayes")
Comparando
```bash
print(y_credit_teste)
```
![Alt text](imgs/bay6.png "Bayes")
Comparando com o sklearn
```bash
print(accuracy_score(y_credit_teste, previsoes))
```
![Alt text](imgs/bay7.png "Bayes")
O algorítmo acertou 93%

Matriz de confusão
```bash
print(confusion_matrix(y_credit_teste, previsoes))
```
![Alt text](imgs/bay8.png "Bayes")
428 pagam e foram classificados como pagam

8 pagam e foram classificados como não pagam

23 clientes não pagam e foram classificados como pagam

41 clientes não pagam e foram classificados como não pagam

Matriz de confusão yellowbrick
```bash
cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
plt.show()
```
![Alt text](imgs/bay9.png "Bayes")
Relatório de classificação
```bash
print(classification_report(y_credit_teste, previsoes))
```
![Alt text](imgs/bay10.png "Bayes")

### Base censo
```bash
with open('census.pkl', mode='rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)
print(X_census_treinamento.shape, y_census_treinamento.shape)
print(X_census_teste.shape, y_census_teste.shape)
```
![Alt text](imgs/bay11.png "Bayes")

Treinando e executando o algorítmo
```bash
naive_census_data = GaussianNB()
naive_census_data.fit(X_census_treinamento, y_census_treinamento)
previsoes = naive_census_data.predict(X_census_teste)
print(previsoes)
print(y_census_teste)
```
![Alt text](imgs/bay12.png "Bayes")
O treinamento errou a maioria das previsões

Acuracidade
```bash
print(accuracy_score(y_census_teste, previsoes))
```
![Alt text](imgs/bay13.png "Bayes")

Taxa muito baixa. O fato de termos apenas duas classes, influencia nesse resultado
Se não executar o escalonamento, pode chegar a 70% de acuracidade, no caso Bayes.

Matriz de confusão
```bash
cm = ConfusionMatrix(naive_census_data)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
plt.show()
```
![Alt text](imgs/bay14.png "Bayes")

Relatório de classificação
```bash
print(classification_report(y_census_teste, previsoes))
```
![Alt text](imgs/bay15.png "Bayes")

## Aprendizagem por árvores de decisão
![Alt text](imgs/tree.png "Tree")
Cálculo da entropia. Verifica o quão organizado estão os dados  
Entropia geral
![Alt text](imgs/entropy.png "Tree")
```bash
Risco
Alto = 6/14  
Moderado = 3/14  
Baixo = 5/14

E(s) = -6/14 * log(6/14;2) - 3/14 * log(3/14;2) - 5/14 * log(5/14;2) = 1,53
```
Como saber o topo da árvore?
Calcular o gain de cada atributo

Cálculo do ganho de informação  
![Alt text](imgs/gain.png "Tree")
```bash
Historia de credito
Boa (5/14) -> Alto (1/5), Moderado (1/5), Baixo (3/5)
Desconhecida (5/14) -> Alto (2/5), Moderado (1/5), Baixo (2/5)
Ruim (4/14) -> Alto (3/4), Moderado (1/4), Baixo (0)

E(s) = -1/5 * log(1/5;2) - 1/5 * log(1/5;2) - 3/5 * log(3/5;2) = 1,37
E(s) = -2/5 * log(2/5;2) - 1/5 * log(1/5;2) - 2/5 * log(2/5;2)= 1,52
E(s) = -3/4 * log(3/4;2) - 1/4 * log(1/4;2) - 0 * log(0;2) = 0,81

Ganho(Historia) = 1,53 - (5/14 * 1,37) - (5/14 * 1,52) - (4/14 * 0,81) = 0,26
```

```bash
Dívida
Alta (5/14) -> Alto (4/7), Moderado (1/7), Baixo (2/7)
Baixa (5/14) -> Alto (2/7), Moderado (2/7), Baixo (3/7)

E(s) = -4/7 * log(4/7;2) - 1/7 * log(1/7;2) - 2/7 * log(2/7;2) = 1,38
E(s) = -2/7 * log(2/7;2) - 2/7 * log(2/7;2) - 3/7 * log(3/7;2)= 1,56

Ganho(Divida) = 1,53 - (7/14 * 1,38) - (7/14 * 1,56) = 0,06
```

```bash
Garantias
Nenhuma (11/14) -> Alto (6/11), Moderado (2/11), Baixo (3/11)
Adequada (3/14) -> Alto (0), Moderado (1/3), Baixo (2/3)

E(s) = -6/11 * log(6/11;2) - 2/11 * log(2/11;2) - 3/11 * log(3/11;2) = 1,44
E(s) = -0 * log(0;2) - 1/3 * log(1/3;2) - 2/3 * log(2/3;2) = 0,92

Ganho(Garantias) = 1,53 - (11/14 * 1,44) - (3/14 * 0,92) = 0,20
```

```bash
Renda Anual
<15     (3/14) -> Alto (3/3), Moderado (0), Baixo (0)
>=15<35 (4/14) -> Alto (2/4), Moderado (2/4), Baixo (0)
>35     (7/14) -> Alto (1/7), Moderado (1/7), Baixo (5/7)

E(s) = -3/3 * log(3/3;2) - 0 * log(0;2) - 0 * log(0;2) = 0,00
E(s) = -2/4 * log(2/4;2) - 2/4 * log(2/4;2) - 0 * log(0;2) = 1,00 
E(s) = -1/7 * log(1/7;2) - 1/7 * log(1/7;2) - 5/7 * log(5/7;2)= 1,15

Ganho(Garantias) = 1,53 - (3/14 * 0) - (4/14 * 1,00) - (7/14 * 1,15) = 0,66
```
```bash
História de crédito = 0.26
Dívida = 0,06
Garantias = 0,20
Renda = 0,66
```
O atributo renda, teve o melhor desempenho, logo, será escolhido como o topo da árvore de decisão  

Vantagens:
* Fácil interpretação
* Não precisa normalização ou normalização
* Rápido para classificar novos registros

Desvantagens:  
* Geração de árvores muito complexas
* Pequenas mudanças nos dados pode mudar a árvore (poda pode ajudar)
* Problema NP-completo para construir a árvore  

Eram muito populares em meados dos anos 90. Hoje temos algorítmos melhores, como redes neurais, SVM, KNN  
O Random Forest melhoram o desempenho  
CART - classification and regression trees  

### Árvores de decisão - Base risco crédito

```bash
with open('risco_credito.pkl', mode='rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)
print(X_risco_credito)
```
![Alt text](imgs/tree2.png "Tree")

```bash
print(y_risco_credito)
```
![Alt text](imgs/tree3.png "Tree")
```bash
print(tree.plot_tree(arvore_risco_credito))
plt.show()
```
```bash
[Text(0.5, 0.9, 'x[3] <= 1.5\nentropy = 1.531\nsamples = 14\nvalue = [6, 5, 3]'), 
Text(0.2, 0.7, 'x[3] <= 0.5\nentropy = 0.863\nsamples = 7\nvalue = [5, 0, 2]'), 
Text(0.35, 0.8, 'True  '), Text(0.1, 0.5, 'entropy = 0.0\nsamples = 3\nvalue = [3, 0, 0]'), 
Text(0.3, 0.5, 'x[0] <= 0.5\nentropy = 1.0\nsamples = 4\nvalue = [2, 0, 2]'), 
Text(0.2, 0.3, 'entropy = 0.0\nsamples = 1\nvalue = [0, 0, 1]'), 
Text(0.4, 0.3, 'x[1] <= 0.5\nentropy = 0.918\nsamples = 3\nvalue = [2, 0, 1]'), 
Text(0.3, 0.1, 'entropy = 0.0\nsamples = 2\nvalue = [2, 0, 0]'), 
Text(0.5, 0.1, 'entropy = 0.0\nsamples = 1\nvalue = [0, 0, 1]'), 
Text(0.8, 0.7, 'x[0] <= 1.5\nentropy = 1.149\nsamples = 7\nvalue = [1, 5, 1]'), 
Text(0.65, 0.8, '  False'), 
Text(0.7, 0.5, 'x[0] <= 0.5\nentropy = 0.65\nsamples = 6\nvalue = [1, 5, 0]'), 
Text(0.6, 0.3, 'entropy = 0.0\nsamples = 3\nvalue = [0, 3, 0]'), 
Text(0.8, 0.3, 'x[2] <= 0.5\nentropy = 0.918\nsamples = 3\nvalue = [1, 2, 0]'), 
Text(0.7, 0.1, 'entropy = 0.0\nsamples = 1\nvalue = [0, 1, 0]'), 
Text(0.9, 0.1, 'entropy = 1.0\nsamples = 2\nvalue = [1, 1, 0]'), 
Text(0.9, 0.5, 'entropy = 0.0\nsamples = 1\nvalue = [0, 0, 1]')]
```
![Alt text](imgs/tree4.png "Tree")
```bash
previsores = ['história','dívida','garantias','renda']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=[10,10])
tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_, filled=True)
plt.show()
```
![Alt text](imgs/tree7.png "Tree")
```bash
previsoes = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes)
```
![Alt text](imgs/tree8.png "Tree")

### Árvores de decisão - Base crédito
```bash
with open('credit.pkl', mode='rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
```
```bash
print(X_credit_treinamento.shape, y_credit_treinamento.shape)
print(X_credit_teste.shape, y_credit_teste.shape)
```
![Alt text](imgs/tree9.png "Tree")
```bash
arvore_credito = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credito.fit(X_credit_treinamento, y_credit_treinamento)
previsoes = arvore_credito.predict(X_credit_teste)
print(previsoes)
```
![Alt text](imgs/tree10.png "Tree")
```bash
print(y_credit_teste)
```
![Alt text](imgs/tree11.png "Tree")
```bash
accuracy = accuracy_score(y_credit_teste, previsoes)
print(accuracy)
```
![Alt text](imgs/tree12.png "Tree")
```bash
cm = ConfusionMatrix(arvore_credito)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
plt.show()
```
![Alt text](imgs/tree13.png "Tree")
```bash
print(classification_report(y_credit_teste, previsoes))
```
![Alt text](imgs/tree14.png "Tree")
```bash
previsores = ['income','age','loan']
class_names = ['0','1']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
class_array = [str(i) for i in arvore_credito.classes_]
print(class_array)
tree.plot_tree(arvore_credito, feature_names=previsores, class_names=class_array, filled=True)
fig.savefig('arvore_credit.png')
plt.show()
```
![Alt text](arvore_credit.png "Tree")

### Árvores de decisão - Base censo
```bash
with open('census.pkl', mode='rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_census.fit(X_census_treinamento, y_census_treinamento)
previsoes = arvore_census.predict(X_census_teste)
print(previsoes)
```
![Alt text](imgs/tree15.png "Tree")
```bash
print(y_census_teste)
```
![Alt text](imgs/tree16.png "Tree")
```bash
accuracy = accuracy_score(y_census_teste, previsoes)
print(accuracy)
```
![Alt text](imgs/tree17.png "Tree")
```bash
cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
plt.show()
```
![Alt text](imgs/tree18.png "Tree")
```bash
print(classification_report(y_census_teste, previsoes))
```
![Alt text](imgs/tree19.png "Tree")

### Random forest
Ensemble learning (aprendizagem em conjunto)
* Usar média (regressão) ou votos da maioria (classificação) para dar a resposta final

Escolhe de forma aleatória k, atributos para comparação da métrica de pureza / impureza (impureza de gini / entropia)

```bash
with open('credit.pkl', mode='rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
random_forest_credit = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)
previsoes = random_forest_credit.predict(X_credit_teste)
print(previsoes)
print(y_credit_teste)
```
![Alt text](imgs/tree20.png "Tree")

```bash
accuracy = accuracy_score(y_credit_teste, previsoes)
print(accuracy)
```
![Alt text](imgs/tree21.png "Tree")

## Aprendizado por regras
Teoria sobre indução de regras - algoritmo OneR e PRISM  
Árvores de decisão x regras  
Regras(CN2) com orange  

### Base de risco de crédito  

```bash
pip install Orange3
import Orange
base_risco_credito = Orange.data.Table('risco_credito_regras.csv')
print(base_risco_credito)
```
![Alt text](imgs/regras1.png "Tree")

```bash
cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)
for regras in regras_risco_credito.rule_list:
    print(regras)
```
![Alt text](imgs/regras2.png "Tree")

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
```bash
previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
print(previsoes)
array([1, 0])

print(base_risco_credito.domain.class_var.values)
('alto', 'baixo', 'moderado')

for i in previsoes:
  #print(i)
  print(base_risco_credito.domain.class_var.values[i])

baixo
alto
```
### Base de crédito data

```bash
base_credit = Orange.data.Table('risco_credito_regras.csv')
print(base_credit.domain)
[income, age, loan | default]

base_dividida = Orange.evaluation.testing.sample(base_credit, n = 0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
print(len(base_treinamento), len(base_teste))
(1500, 500)

cn2 = Orange.classification.rules.CN2Learner()
regras_credit = cn2(base_treinamento)
for regras in regras_credit.rule_list:
  print(regras)

IF age>=34.9257164876908 THEN default=0 
IF loan<=2507.64970973955 AND income>=20145.9885970689 THEN default=0 
IF income<=31702.3342987522 AND loan>=3665.88089899456 THEN default=1 
IF loan>=7660.34617051509 AND loan>=9601.375482171099 THEN default=1 
IF loan>=7660.34617051509 AND loan>=9595.28628892989 THEN default=0 
IF loan>=7660.34617051509 AND age>=21.4227129220963 THEN default=1 
IF income<=22925.8120805025 AND age>=34.7410444347188 THEN default=0 
IF income<=24857.6948815025 AND age>=26.382710393052502 THEN default=1 
IF loan<=5898.0008931423 AND income>=44324.286368209905 THEN default=0 
IF age>=34.851817262359 THEN default=0 
IF age>=34.5964918844402 THEN default=1 
IF age>=33.8957485635765 THEN default=0 
IF age>=33.6895613595843 THEN default=1 
IF age>=33.2456503002904 THEN default=0 
IF loan<=4859.2352866332 AND age>=32.6760442531668 THEN default=1 
IF age>=33.0920195085928 THEN default=1 
IF loan<=4859.2352866332 AND income>=34667.0204005204 THEN default=0 
IF income<=57787.565658800304 AND loan>=6342.56790924236 THEN default=1 
IF loan>=8092.982779946221 THEN default=1 
IF income>=52841.5164374746 AND income>=58828.2921165228 THEN default=0 
IF loan<=4859.2352866332 AND age>=21.598676353692998 THEN default=0 
IF age>=24.4157260977663 AND age>=26.854012909811 THEN default=1 
IF income>=52841.5164374746 THEN default=0 
IF loan>=5968.4420381087 THEN default=1 
IF income>=42522.5757574663 THEN default=0 
IF income>=34635.7447491027 THEN default=1 
IF income>=32197.6207010448 THEN default=0 
IF income<=25146.5956843458 AND age>=21.3656869572587 THEN default=1 
IF income>=26218.4948474169 THEN default=1 
IF income<=25146.5956843458 THEN default=0 
IF TRUE THEN default=0 

previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])
print(previsoes)
<Orange.evaluation.testing.Results at 0x7f85496cf670>

print(Orange.evaluation.CA(previsoes))
array([0.964])
```

##







