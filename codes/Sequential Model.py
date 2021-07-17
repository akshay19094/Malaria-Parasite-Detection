import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

infected = []
#read parasitized images
for image in os.listdir('cell_images/Parasitized/'):
    if image.endswith(".png"):
        infected.append(image)
        
uninfected = []
#read unifected images
for image in os.listdir('cell_images/Uninfected/'):
    if image.endswith(".png"):
        uninfected.append(image)
        
print(len(infected))
print(len(uninfected))

data=[]
label=[]

print("Infected")
for i,img in enumerate(infected):
    print(i)
    kernel = np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])
    image=cv2.imread('cell_images/Parasitized/{}'.format(img))
    #resize image to 75*75*3
    resized_image=cv2.resize(image,(75,75))
    #apply convolution of filter and image
    image=cv2.filter2D(resized_image,-1,kernel)
    #transform image from RGB space to YUV space
    YUV_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #equalize histogram
    YUV_image[: ,: , 0] = cv2.equalizeHist(YUV_image[:,:,0])
    image=cv2.cvtColor(YUV_image, cv2.COLOR_YUV2RGB)
    #transform image from YUV space to RGB space
    data.append(np.array(image))
    label.append(0)
    #plt.imshow(image)
    #break
    
print("Uninfected")
for i,img in enumerate(uninfected):
    print(i)
    kernel = np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])
    image=cv2.imread('cell_images/Uninfected/{}'.format(img))
    #resize image to 75*75*3
    resized_image=cv2.resize(image,(75.75))
    #apply convolution of filter and image
    image=cv2.filter2D(resized_image,-1,kernel)
    #transform image from RGB space to YUV space
    YUV_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #equalize histogram
    YUV_image[: ,: , 0] = cv2.equalizeHist(YUV_image[:,:,0])
    #transform image from YUV space to RGB space
    image=cv2.cvtColor(YUV_image, cv2.COLOR_YUV2RGB)
    data.append(np.array(image))
    label.append(1)
    #plt.imshow(image)

cells=np.array(data)
labels=np.array(label)
print(cells.shape,labels.shape)

#Normalize RGB components to values between 0 and 1
cells=cells.astype(np.float32)/255
labels=labels.astype(np.int32)

#obtain the ID of each image
n = np.arange(cells.shape[0])

#shuffle images
np.random.shuffle(n)

cells=cells[n]
labels=labels[n]

import keras
#70-30 train test split
x_train, x_test, y_train, y_test = train_test_split(cells,labels,test_size=0.3,random_state=1)

#convert labels to categorical form
y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

import tensorflow as tf
from keras import datasets, layers, models
model = models.Sequential()

#Three Convolution 2D layers added to the model
model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu',input_shape=(75,75,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

model.add(layers.Flatten())

#Dense relU and softmax layer added to the model
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2,activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting model over the entire dataset with batch size 50 and number of epochs 20
model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)

#print accuracy
accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Test Accuracy= {}'.format(accuracy[1]))


