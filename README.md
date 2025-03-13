# Machine Learn

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