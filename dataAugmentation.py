import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator as idg
from skimage import exposure, color
import cv2

# helper function to help identify datatypes
def detail(x,val):
    try:
        print(type(x))
        print(x.dtype)
        print(x.shape)
    except:
        print(type(x))
    if val:
        print(x)

# Reading in the data and setting it up
images=np.load('train_and_test.npz')
data=[images[key] for key in images]
completeX=np.load('evenDataTrainX.npy')
completeY=np.load('evenDataTrainY.npy')

# Splitting up the data into x, y, and test
x,y,test=data[0],np_utils.to_categorical(data[1]),data[2]
num_class=y.shape[1]

# Splitting up the data based on the classification
classIndexList=[]
for i in range(num_class):
    classIndexList.append(np.argwhere(data[1]==i))

# adding more samples to even out the histogram
newX=x
tempY=data[1]
newY=data[1]
for l in classIndexList:
    start,end=l[0][0],l[len(l)-1][0]
    dist=4000-len(l)
    shift=0.2
    datagen = idg(width_shift_range=shift, height_shift_range=shift)
    datagen.fit(x[start:end])
    loops=int(dist/len(l))
    remains=dist%len(l)
    i=0
    print(len(l))
    print(dist)
    print(loops)
    print(remains)
    for batch in datagen.flow(x[start:end],tempY[start:end],batch_size=dist):
        for t in range(loops):
            newX=np.append(newX,batch[0].astype('uint8'),axis=0)
            newY=np.append(newY,batch[1].astype('uint8'),axis=0)
        break
        
    for batch in datagen.flow(x[start:end],tempY[start:end],batch_size=remains):
        newX=np.append(newX,batch[0].astype('uint8'),axis=0)
        newY=np.append(newY,batch[1].astype('uint8'),axis=0)
        break