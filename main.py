from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50
import cv2
import os

tipos = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','Normal_Videos_event','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']



def modelo(output = 14):
    modelo_base_RES = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model_head = modelo_base_RES.output
    model_head = AveragePooling2D(pool_size=(7, 7))(model_head)
    model_head = Flatten(name="flatten")(model_head)
    model_head = Dense(512, activation="relu")(model_head)
    model_head = Dropout(0.5)(model_head)
    model_head = Dense(output, activation="softmax")(model_head)
    model = Model(inputs=modelo_base_RES.input,outputs=model_head)
    return model

x = modelo()
