import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator as idg
from skimage import exposure, color
import cv2

x=np.load('greyscaleImgTrain.npy')
y=np_utils.to_categorical(np.load('evenDataTrainY.npy'))
c = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
np.random.shuffle(c)
xshuffle=c[:, :x.size//len(x)].reshape(x.shape)
yshuffle=c[:, :y.size//len(x)].reshape(y.shape)

# #Building of the nn
model_conv=Sequential()

model_conv.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(32, 32,1)))

model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(BatchNormalization())

model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(BatchNormalization())

model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(BatchNormalization())
model_conv.add(Flatten())

model_conv.add(Dense(128, activation='relu'))

model_conv.add(Dense(43, activation='softmax'))

model_conv.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

print(model_conv.summary())

model_conv.fit(xshuffle[:100000], yshuffle[:100000], 
	validation_data=(xshuffle[100000:], yshuffle[100000:]), epochs=100, batch_size=1000)
model_conv.save('ConModel')
