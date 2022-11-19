import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from progress.bar import FillingSquaresBar
from imgaug import augmenters as iaa
from tensorflow.keras import layers
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from email import header
import tensorflow as tf
from time import sleep
import pandas as pd
import numpy as np
import random
import glob
import cv2
import os


def get_center_names(filepath):

        return filepath.split('\\')[-1]


def read_csv(path):
        columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
        data    = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)
        data['Center'] =data['Center'].apply(get_center_names)
        # Center_column  = data['Center'].apply(get_center_names)
        # Steering_column = data['Steering']

        return data  #, Center_column, Steering_column


def laod_data(path, data):
        imagepath = []
        steering  = []

        for i in range(len(data)):
                indexedData = data.iloc[i]
                imagepath.append(os.path.join(path, 'IMG', indexedData[0]))
                steering.append(float(indexedData[3]))

        imagepath = np.asarray(imagepath)
        steering = np.asarray(steering)

        return imagepath, steering


def drow_histogram(data, display = True):

        nbins = 31
        samplesperbin = 500
        # hist, bins = np.histogram(data['Steering'], nbins)
        if display:
                plt.hist(data['Steering'], bins = 31)
                plt.show()


def augmentImage(imgPath,steering):

    img = mpimg.imread(imgPath)


    #PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent = {'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale = (1,1.2))
        img = zoom.augment_image(img)
    
    #BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    #FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering

    return img, steering

def preprocessing(img):

        img = img[60:135, :,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.resize(img, (200,66))
        img = img/255
        return img


def batchGen(imagespath, steeringList,batchSize,trainFlag):
        while True:

                imgBatch = []
                steeringBatch = []

                for i in range(batchSize):
                        index = random.randint(0, len(imagespath)-1)
                        if trainFlag:
                                img, steering = augmentImage(imagespath[index], steeringList[index])
                        else:
                                img = mpimg.imread(imagespath[index])
                                steering = steeringList[index]
                        img = preprocessing(img)
                        imgBatch.append(img)
                        steeringBatch.append(steering)

                yield (np.asarray(imgBatch), np.asarray(steeringBatch))

def createmodel():
        model = Sequential()
        model.add(layers.Conv2D(24, (5,5), (2,2), input_shape = (66,200,3), activation = 'elu'))
        model.add(layers.Conv2D(36, (5,5), (2,2), activation = 'elu'))
        model.add(layers.Conv2D(48, (5,5), (2,2), activation = 'elu'))
        model.add(layers.Conv2D(64, (3,3), activation = 'elu'))
        model.add(layers.Conv2D(64, (3,3), activation = 'elu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(100,activation = 'elu'))
        model.add(layers.Dense(50,activation = 'elu'))
        model.add(layers.Dense(10,activation = 'elu'))
        model.add(layers.Dense(1))

        model.compile(Adam(learning_rate = 0.0001), loss = 'mse')

        return model