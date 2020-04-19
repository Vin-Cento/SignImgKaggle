import cv2
from scipy import misc, ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure, color


# modifing the image with contrast adjustingment and smoothing the images
completeX=np.load('evenDataTrainX.npy')

# Reading in the data and setting it up
images=np.load('train_and_test.npz')
data=[images[key] for key in images]

# Splitting up the original data into Trainx, Trainy, and test
originalX,originalY,test=data[0],data[1],data[2]

cleanImgTrain=np.zeros((completeX.shape[0],32,32,3),dtype=np.float64)
cleanImgTest=np.zeros((test.shape[0],32,32,3))

for i in range(completeX.shape[0]):
    cleanImgTrain[i]=cv2.GaussianBlur(exposure.equalize_hist(completeX[i]),(3,3),0)
np.save('finishTrain',cleanImgTrain)
print('first one')
for i in range(test.shape[0]):
	cleanImgTest[i]=cv2.GaussianBlur(exposure.equalize_hist(test[i]),(3,3),0)
np.save('finishTest',cleanImgTest)
