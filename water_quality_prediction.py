# -*- coding: utf-8 -*-
"""Water_Quality_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/salsabilar311/Predictive-Analytics_Water-Quality/blob/main/Water_Quality_Prediction.ipynb
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/salsabilar311/Predictive-Analytics_Water-Quality/main/water_potability.csv?token=GHSAT0AAAAAACNPNJMQZVVQDU4EUO5ADYQCZN3EHTA')
df

"""**Remove Missing Value**"""

df = df.dropna()

"""**Remove Outlier**"""

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
df = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

df

"""**View feature correlation with potability**"""

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik", size=20)

"""**Split Data**"""

X = df.drop(["Potability"],axis =1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

"""**Normalized X_train**"""

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train

"""**K-Nearest Neighbor**"""

acc = []
model = []

# Metode KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

# Memprediksi hasil Test Set
y_pred_KNN = knn.predict(X_test)

# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)

# Menghitung nilai akurasi dari klasifikasi naive bayes
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_KNN))

acc.append((cm_KNN[0][0]+cm_KNN[1][1])/(cm_KNN[0][0]+cm_KNN[1][1]+cm_KNN[0][1]+cm_KNN[1][0]))
model.append('KNN')

"""**Random Forest**"""

# Metode Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Memprediksi hasil Test Set
y_pred_rfc = rfc.predict(X_test)

# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rfc = confusion_matrix(y_test, y_pred_rfc)

# Menghitung nilai akurasi dari klasifikasi naive bayes
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rfc))

acc.append((cm_rfc[0][0]+cm_rfc[1][1])/(cm_rfc[0][0]+cm_rfc[1][1]+cm_rfc[0][1]+cm_rfc[1][0]))
model.append('Random Forest')

"""**Naive Bayes**"""

# Metode Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Memprediksi hasil Test Set
y_pred_gnb = gnb.predict(X_test)

# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

# Menghitung nilai akurasi dari klasifikasi naive bayes
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_gnb))

acc.append((cm_gnb[0][0]+cm_gnb[1][1])/(cm_gnb[0][0]+cm_gnb[1][1]+cm_gnb[0][1]+cm_gnb[1][0]))
model.append('Naive Bayes')

"""**Model Evaluation**"""

df_model = pd.DataFrame(data=[acc], columns=model)
df_model

"""# Conclusion
Random forest adalah algoritma yang memiliki akurasi yang tertinggi dari ketiga model. Dengan nilai akurasi **0.58**. Oleh karena itu untuk memprediksi kualitas air akan digunakan Random Forest sebagai model machine learning.
"""