import tensorflow as tf
from keras import layers, models
import os
import numpy as np
import cv2
import random

#Parametros 

width = 300
height = 300 
ruta_train = '' #Ruta de las imagenes de entrenamiento
ruta_predict = '' #Ruta de las imagenes prediccion

train_x = []
train_y = []

for i in os.listdir(ruta_train):
    for j in os.listdir*(ruta_train + i):
        img = cv2.imread(ruta_train+i+'/'+j)
        resized_image = cv2.resize(img, (width, height))

        train_x.append(resized_image)

        if i == '': #Imagen a buscar
            train_y.append([0,1]) # La lista aumenta de tamaño segun como se quieran clasificar los datos
        else:
            train_y.append([1,0])

x_data = np.array(train_x)
y_data = np.array(train_y)

#Red Neuronal Base
model = tf.keras.Sequential([
    layers.Conv2D(32, 3,3, input_shape=(width, height, 3)), #Numero de filtros (32) y numeros de pixeles (3x3)
    layers.Activation('relu'), #Estructuracion de datos
    layers.MaxPooling2D(pool_size=(2,2)), # Reducir la seccion de busqueda para buscar mas patrones
    layers.Conv2D(32, 3,3), #Numero de filtros (32) y numeros de pixeles (3x3)
    layers.Activation('relu'), #Estructuracion de datos
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(32, 3,3), #Numero de filtros (32) y numeros de pixeles (3x3)
    layers.Activation('relu'), #Estructuracion de datos
    layers.MaxPooling2D(pool_size=(2,2)), 
    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0,5),
    layers.Dense(2), #Numero de neuronas de vido a la cantidad de datos de la lista
    layers.Activation('sigmoid'), #Capa de probabilidades entre 0 y 1
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 100 # Numero de veces que el modelo revisa los datos 

model.fit(x_data, y_data, epochs = epochs)

models.save_model(model, 'Dr.keras')
