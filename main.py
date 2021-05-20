from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import cv2
from imutils import paths
import numpy as np


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


def array_processing(imagens_use):
    array_process = np.array(imagens_use)
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
                count += 1 # i.e. at 15 fps, this advances one second
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


def training_data_load_reshape():
    dados_treino =np.load('training_data.npy')
    frame = dados_treino.shape[2]
    frame = frame-frame%10
    dados_treino = dados_treino[:,:,:frame]
    dados_treino = dados_treino.reshape(-1,227,227,10)
    dados_treino = np.expand_dims(dados_treino,axis=4)
    dados_alvo = dados_treino.copy()
    return (dados_treino,dados_alvo)

dados_train,dados_tgt = training_data_load_reshape()

sek = modelo()

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

sek.fit(dados_train,dados_tgt, batch_size=1,epochs=6,callbacks = [callback_early_stopping],verbose=1, use_multiprocessing=True)

sek.save('model')