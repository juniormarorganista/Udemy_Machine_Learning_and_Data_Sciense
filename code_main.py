import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

base_credit = pd.read_csv('Bases_de_dados/credit_data.csv')

#print('\n print all \n\n',base_credit.head(10))
#print('\n print 10 primeiros \n\n',base_credit.head(10))
#print('\n print 10 ultimos \n\n',base_credit.tail(10))

#print(base_credit.describe()) # estatistica basica

#print(base_credit[base_credit['income'] >= 69995.685578])
#print(base_credit[base_credit['loan'] <= 1.377630])
#print(base_credit[base_credit['loan'] <= 1.377630])

### plot datas
#sns.countplot(x=base_credit['default']);
#plt.show()
#sns.countplot(x=base_credit['age']);
#plt.show()
#plt.hist(x=base_credit['age']);
#plt.show()
#plt.hist(x=base_credit['income']);
#plt.show()
#plt.hist(x=base_credit['loan']);
#plt.show()
#grafico = px.scatter_matrix(base_credit,dimensions=['age','income','loan'],color='default')
#grafico.show()

#base_credit.loc[base_credit['age'] < 0]

# Apagar a coluna quando for o melhor
base_credit2 = base_credit.drop('age',axis=1)

# Apaga somente os dados pelo filtro quando for o melhor
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

# Preencher os valores inconsistente com a média (valores da idade maior que zero - dado consistente)
base_credit['age'][base_credit['age']>0].mean()

# Preencher os valores inconsistente com a média (valores da idade maior que zero - dado consistente)
base_credit['age'][base_credit['age']>0].mean()

# Preencher os valores inconsistente com a média (valores da idade maior que zero - dado consistente)
# < 0,'age' ---> para trocar somente a idade incorreta
base_credit.loc[base_credit['age'] < 0,'age'] = base_credit['age'][base_credit['age']>0].mean()

base_credit.isnull().sum()

base_credit.loc[pd.isnull(base_credit['age'])]

base_credit['age'].fillna(base_credit['age'].mean(),inplace=True) 

base_credit.isnull().sum()

base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]

base_credit.loc[base_credit['clientid'].isin([29,31,32])]

X_credit = base_credit.iloc[:,1:4].values

Y_credit = base_credit.iloc[:,4].values

type(X_credit)

type(Y_credit)

X_credit[:,0].min(),X_credit[:,1].min(),X_credit[:,2].min()

# Deixa os dados na mesma escala
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

X_credit[:,0].min(),X_credit[:,1].min(),X_credit[:,2].min()

base_census = pd.read_csv('Bases_de_dados/census.csv')

base_census.describe()

np.unique(base_census['income'], return_counts=True)

#sns.countplot(x=base_census['income'])
plt.hist(x=base_census['age'])
plt.hist(x=base_census['education-num'])
plt.hist(x=base_census['hour-per-week'])
grafico = px.treemap(base_census,path=['workclass','age'])
grafico.show()






















