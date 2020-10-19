import numpy as np #crear vectores y matrices grandes multidimensionales
import pandas as pd # fusionar y unir datos, leer archivos csv
#metaestimador que se ajusta a varios clasificadores de árboles de decisión en varias submuestras del conjunto de datos
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
#Modelo para Regresion Logistica
from sklearn import linear_model
# implementa funciones que evalúan el error de predicción para propósitos específicos
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import logo

logo.Banner_Presentacion()

#//////////////////////////////////////////////////////////////////////////////////

traindata = pd.read_csv('kddtrain.csv')
testdata = pd.read_csv('kddtest.csv')

#Indexamos la ubicacion de los nuemeros enteros para seleccionar su posicion
X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

scaler = Normalizer().fit(X) #normalizamos hasta escalar datos con 1 y 0; lo entrenamos
trainX = scaler.transform(X) #transformamos los valores con X 

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)

#//////////////////////////////////////////////////////////////////////////////////
#Modelo para Regresion Logistica
model = LogisticRegression()
model.fit(traindata, trainlabel)

# hacer la prediccion
expected = testlabel
predicted = model.predict(testdata) #la prediccion nos dara 0 o 1 como salida
proba = model.predict_proba(testdata) #predecimos las probabilidades que sean 1

y_train1 = expected
y_pred = predicted

#calcula la exactitud del conjunto
exactitud = accuracy_score(y_train1, y_pred) 
#calcula la recuperacion de valores verdaderos positivos y falsos negativos
recuperacion = recall_score(y_train1, y_pred , average="binary") 
#relacion de verdaderos positivos y falsos positivos
precision = precision_score(y_train1, y_pred , average="binary") 
#promedio ponderado de la precision y recuperacion, donde 1 es la mejor puntuacion y 0 la peor
f1 = f1_score(y_train1, y_pred, average="binary") 

print("Regresion Logistica")
print("Exactitud: ",exactitud)
print("Precision: ",precision)
print("Recuperacion: ",recuperacion)
print("Puntuacion F1: ",f1)

#//////////////////////////////////////////////////////////////////////////////////

#Modelo para Naive BAYES
model = GaussianNB()
model.fit(traindata, trainlabel)
#print(model)

# Prediccion
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)


y_train1 = expected
y_pred = predicted

exactitud = accuracy_score(y_train1, y_pred)
recuperacion = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print(" Naive Bayes ")
print("Exactitud: ",exactitud)
print("Precision: ",precision)
print("Recuperacion: ",recuperacion)
print("Puntuacion F1: ",f1)

#//////////////////////////////////////////////////////////////////////////////////

#Modelo para un Clasificador de árbol de decisión
model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
#print(model)

# Realizamos la prediccion
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)


# resumir el ajuste del modelo
y_train1 = expected
y_pred = predicted
exactitud = accuracy_score(y_train1, y_pred)
recuperacion = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("Clasificador de árbol de decisión")
print("Exactitud: ",exactitud)
print("Precision: ",precision)
print("Recuperacion: ",recuperacion)
print("Puntuacion F1: ",f1)