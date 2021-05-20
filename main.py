from pandas.core import frame
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50
import cv2
import os
from imutils import paths
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.python.ops.numpy_ops.np_array_ops import array


def prep(path):
    imagem = cv2.resize(img_to_array(load_img(path)), (227,227), interpolation= cv2.INTER_AREA)
    rgb_cinza = 0.2989*imagem[:,:,0]+0.5870*imagem[:,:,1]+0.1140*imagem[:,:,2]
    imagens_cinza.append(rgb_cinza)

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

def array_processing(array_process):
    array_process = np.array(imagens_cinza)
    x,y,z = array_process.shape
    array_process.resize(y,z,x)
    array_process = np.clip((array_process-array_process.mean())/array_process.std(),0,1)
    np.save('training_data.npy',array_process)

def video_to_frame(files):
    # Gets one frame every t

    for video in files:
        cap = cv2.VideoCapture(video)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                cami = video.split(sep='\\')[1].replace('.avi','')+'frame'+str(count)+'.jpg'
                cv2.imwrite(cami, frame)
                count += 2 # i.e. at 15 fps, this advances one second
                cap.set(1, count)
            else:
                cap.release()
                break

imagens_cinza = []
pathing = list(paths.list_files('training_videos'))
image_list = list(paths.list_images('training_frames/'))

for x in image_list:
    prep(x)

array_processing(imagens_cinza)



