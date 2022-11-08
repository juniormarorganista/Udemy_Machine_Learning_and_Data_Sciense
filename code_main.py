import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

print('Importando base de credito...')
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

#print(base_credit.isnull().sum())
#print(base_credit.loc[pd.isnull(base_credit['age'])])
#print(base_credit['age'].fillna(base_credit['age'].mean(),inplace=True))
#print(base_credit.isnull().sum())

base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]

base_credit.loc[base_credit['clientid'].isin([29,31,32])]

X_credit = base_credit.iloc[:,1:4].values

Y_credit = base_credit.iloc[:,4].values

#print(type(X_credit))
#print(type(Y_credit))
#print(X_credit[:,0].min(),X_credit[:,1].min(),X_credit[:,2].min())

# Deixa os dados na mesma escala
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

X_credit[:,0].min(),X_credit[:,1].min(),X_credit[:,2].min()

print('Importando base de census...')
base_census = pd.read_csv('Bases_de_dados/census.csv')

base_census.describe()

np.unique(base_census['income'], return_counts=True)

#sns.countplot(x=base_census['income'])
#plt.hist(x=base_census['age'])
#plt.hist(x=base_census['education-num'])
#plt.hist(x=base_census['hour-per-week'])
#grafico = px.treemap(base_census,path=['workclass','age'])
#grafico.show()
#grafico = px.treemap(base_census,path=['occupation','relationship','age'])
#grafico.show()
#grafico =px.parallel_categories(base_census,dimensions=['occupation','relationship'])
#grafico.show()
#grafico =px.parallel_categories(base_census,dimensions=['workclass','occupation','relationship'])
#grafico.show()

print(base_census.columns)

X_census = base_census.iloc[:,0:14].values
Y_census = base_census.iloc[:,14].values

#print(Y_census)

label_encoder_workclass    = LabelEncoder()
label_encoder_education    = LabelEncoder()
label_encoder_marital      = LabelEncoder()
label_encoder_occupation   = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race         = LabelEncoder()
label_encoder_sex          = LabelEncoder()
label_encoder_country      = LabelEncoder()

X_census[:,1]  = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3]  = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5]  = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6]  = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7]  = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8]  = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9]  = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

#print(X_census)
#print(len(np.unique(base_census['occupation'])))

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),[1,3,5,6,7,8,9,13])],remainder='passthrough') 

X_census = onehotencoder_census.fit_transform(X_census).toarray()

print(X_census)
print(X_census.shape)

scaler_census = StandardScaler()
X_census      = scaler_census.fit_transform(X_census)

print(X_census)

X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit,Y_credit,test_size=0.25,random_state=0)

print(X_credit_teste.shape,Y_credit_teste.shape)

X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = train_test_split(X_census,Y_census,test_size=0.15,random_state=0)

X_census_teste.shape,Y_census_teste.shape

with open('credit.pkl',mode='wb') as f:
  pickle.dump([X_credit_treinamento, Y_credit_treinamento,X_credit_teste, Y_credit_teste],f)

with open('census.pkl',mode='wb') as f:
  pickle.dump([X_census_treinamento, Y_census_treinamento,X_census_teste, Y_census_teste],f)






























