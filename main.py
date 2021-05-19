from pandas.core import frame
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.layers import Dropout, Conv3D, ConvLSTM2D, Conv3DTranspose, Input, AveragePooling2D
from tensorflow.keras.models import Model,Sequential
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

def modelo():
    model=Sequential()
    model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
    model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
    model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
    model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
    model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))
    model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
    model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    return model

qt6 = modelo()

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