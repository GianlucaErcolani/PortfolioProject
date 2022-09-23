# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 17:35:50 2022

@author: gianl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#data import and exploration

df = pd. read_csv (r"C:\Users\gianl\Downloads\heart_disease_data.csv")

df.head()

df.tail()

df.shape

df.info()

df.isnull().sum()

df.describe()

df['target'].value_counts()

df.columns

#Generating categorical columns values
#cp - chest_pain_type
df.loc[df['cp'] == 0, 'cp'] = 'asymptomatic'
df.loc[df['cp'] == 1, 'cp'] = 'atypical angina'
df.loc[df['cp'] == 2, 'cp'] = 'non-anginal pain'
df.loc[df['cp'] == 3, 'cp'] = 'typical angina'

#restecg - rest_ecg_type
df.loc[df['restecg'] == 0, 'restecg'] = 'left ventricular hypertrophy'
df.loc[df['restecg'] == 1, 'restecg'] = 'normal'
df.loc[df['restecg'] == 2, 'restecg'] = 'ST-T wave abnormality'

#slope - st_slope_type
df.loc[df['slope'] == 0, 'slope'] = 'downsloping'
df.loc[df['slope'] == 1, 'slope'] = 'flat'
df.loc[df['slope'] == 2, 'slope'] = 'upsloping'

#thal - thalassemia_type
df.loc[df['thal'] == 0, 'thal'] = 'nothing'
df.loc[df['thal'] == 1, 'thal'] = 'fixed defect'
df.loc[df['thal'] == 2, 'thal'] = 'normal'
df.loc[df['thal'] == 3, 'thal'] = 'reversable defect'



#splitting the features and target

X = df.drop('target', axis = 1)

y = df['target']

# look at numeric and categorical values separately 
X_cat=df[['sex', 'cp', 'fbs', 'restecg','exang', 'slope', 'thal']]

#scaling
scaler = MinMaxScaler()
df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca' ]] = scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca' ]])
X_num=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca' ]]

#data visualization
for i in X_num.columns:
    plt.hist(X_num[i])
    plt.title(i)
    plt.show()

print(X_cat)
#dummy variables

X_cat1 = pd.get_dummies(X_cat, drop_first=True)
print(X_cat1)


frames = [X_num, X_cat1]

result = pd.concat(frames, axis=1)
print(result)



X1 = result


#Splitting the Data into Training data & Test Data
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)

#Logistic regression
from sklearn.model_selection import cross_val_score


lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X1_train,y_train,cv=5)
print(cv)
print(cv.mean())






