# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:47:42 2017

@author: w2764  D:/windows_sim/driving_log.csv  'D:/Users/w2764/download/data/data/IMG/'+
"""

import csv
import cv2
import numpy as np

lines =[]
with open('D:/windows_sim/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
print(len(lines))
for line in lines:
    for j in range(1):
        source_path = line[j]
        filename = source_path.split('/')[-1]
        current_path = filename
        print(current_path)
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        steering_center =float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "..." # fill in the path to your training IMG directory
        img_center = cv2.imread(line[0].split('/')[-1])
        img_left = cv2.imread(line[1].split('/')[-1])
        img_right = cv2.imread(line[2].split('/')[-1])

        # add images and angles to data set
        images.extend([img_center, img_left, img_right])
        measurements.extend([steering_center, steering_left, steering_right])
       # cv2.imshow('sss1',image)
        #cv2.waitKey(0)
        
        img_center1 = np.fliplr(img_center)
        img_left1 = np.fliplr(img_left)
        img_right1 = np.fliplr(img_right)
        #cv2.imshow('sss',image_flipped)
           #cv2.waitKey(0)
       
        images.extend([img_center1, img_left1, img_right1])
        measurements.extend([-steering_center, -steering_left, -steering_right])
        
X_train0 = np.array(images)
y_train0 = np.array(measurements)




print(X_train0.shape)
print(y_train0.shape)


index =np.arange( X_train0.shape[0])

np.random.shuffle(index)
X_train = X_train0[index[0:int(X_train0.shape[0] * 0.8)]]
y_train = y_train0[index[0:int(X_train0.shape[0] * 0.8)]]


X_test = X_train0[index[int(X_train0.shape[0] * 0.8) +1 :int(X_train0.shape[0])]]
y_test = y_train0[index[int(X_train0.shape[0] * 0.8) +1 :int(X_train0.shape[0])] ]



from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD , Adam
import keras.regularizers as regularizers
#from keras import backend as K
import keras
#import matplotlib.pyplot as plt


num_classes =1
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Lambda(lambda x: (x/255.0 -0.5), input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,24), (0,0))))

#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))


#model.add(Convolution2D(3, 7, 7,subsample =(2,2),activation ='relu' ))


model.add(Conv2D(24, (5, 5),strides =(2,2),activation ='relu'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Conv2D(36, (5, 5),strides =(2,2),activation ='relu' ))
model.add(Conv2D(48, (5, 5),strides =(2,2),activation ='relu'))


model.add(Conv2D(64, (3, 3),strides =(1,1),activation ='relu' ))

model.add(Conv2D(64, (3, 3),strides =(1,1),activation ='relu' ))



model.add(Flatten())  
model.add(Dense(100 ))

#model.add(Activation('relu')), kernel_regularizer=regularizers.l2(1)
model.add(Dense(50))
model.add(Dense(10))
#model.add(Activation('relu'))
model.add(Dense(num_classes ))

#model.add(Dense(1,activation= 'softmax'))
#model.add(Activation('softmax'))
model.summary()

#adam = Adam(lr=0.001, epsilon=1e-08)
model.compile(loss='mse', optimizer = 'adam', metrics = ['accuracy'])

history_object = model.fit(X_train, y_train,epochs =10 ,validation_data=(X_test,y_test),  batch_size = 32,shuffle = True)

model.save('D:/windows_sim/model.h5')

'''
cropping_output = K.function([model.layers[1].input],[model.layers[1].output])
cropped_image = cropping_output([ X_train[1:2,:,:,:]])[0]
print(cropped_image.shape)

plt.imshow( cropped_image[0,:,:,:])

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
#exit()






























