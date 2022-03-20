from keras import models
from keras import callbacks
from keras.engine import input_layer
from keras_preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob
import tensorflow
import scipy


# Number of the Elements in Specific Cases (0-LGG 1-HGG)

ROOT_DIR = "145"
number_of_images = {}
for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
number_of_images.items()
#len(os.listdir("145"))

def dataFolder(p,split):

#########################
# Creating Train Folder #
#########################

    if not os.path.exists("./"+p):
        os.mkdir("./"+p)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs("./"+p+"/"+dir)
            for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR,dir)),size=(math.floor(split*number_of_images[dir])-2),replace=False):
                O=os.path.join(ROOT_DIR,dir,img)
                D=os.path.join("./"+p,dir)
                shutil.copy(O,D)
                os.remove(O)
    else:
        print(f"{p}Folder exists")
dataFolder("train",0.7)
dataFolder("val",0.15)
dataFolder("test",0.15)

#############
#MODEL BUILD#
#############

from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator 
import keras

#Convolution Neural Networks Model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape =(240,240,3)))
model.add(MaxPool2D(pool_size=(2,2))) #removing unnecessary features

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2))) #removing unnecessary features

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2))) #removing unnecessary features

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2))) #removing unnecessary features

model.add(Dropout(rate=0.25)) # to avoid overfitting

model.add(Flatten()) #reduce to 1 dimension
model.add(Dense(units=64,activation='relu')) # multilayer perceptron with 64 outputs
model.add(Dropout(rate=0.25)) # to avoid overfitting
model.add(Dense(units=16,activation='relu')) # multilayer perceptron with 64 outputs
model.add(Dropout(rate=0.25)) # to avoid overfitting
model.add(Dense(units=1,activation='sigmoid')) # multilayer perceptron with 1 output
model.summary()
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
print(model.get_weights())

####################
# Data Preparation #
####################

from keras.preprocessing.image import ImageDataGenerator
def preprocessingImages1(path):
    """
    input: Path
    output: Preprocessed Images
    """

    image_data=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,rescale=1/255,horizontal_flip=True) # data augmentation
    image=image_data.flow_from_directory(directory=path,target_size=(240,240),batch_size=16,class_mode='binary')
    return image

path="train"
train_data=preprocessingImages1(path)
train_data.class_indices
def preprocessingImages2(path):
    """
    input: Path
    output: Preprocessed Images
    """

    image_data=ImageDataGenerator(rescale=1/255)
    image=image_data.flow_from_directory(directory=path,target_size=(240,240),batch_size=16,class_mode='binary')
    return image
path="test"
test_data=preprocessingImages2(path)
path="val"
val_data=preprocessingImages2(path)

###########################################
# Early Stop and Checking Model Situation #
###########################################

from keras.callbacks import ModelCheckpoint,EarlyStopping

# Early Stopping #

es=EarlyStopping(monitor="val_accuracy",min_delta=0.001,patience=10,verbose=1,mode='auto')

# Early Stopping Checkpoint #

mc=ModelCheckpoint(monitor="val_accuracy",filepath="./bestmodel.h5",verbose=1,save_best_only=True,mode='auto')
cd=[es,mc]

# Model Training #

hs= model.fit(x=train_data,steps_per_epoch=5,epochs=30,verbose=1,validation_data=val_data,validation_steps=16,callbacks=cd)

# Model Graphical Transformation #

h=hs.history
h.keys()

# Accuracy Graphing #

import matplotlib.pyplot as plt
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c="red")
plt.title("accuracy(blue) vs validation accuracy(red)")
plt.show()

# Loss Graphing #

plt.plot(h['loss'])
plt.plot(h['val_loss'], c="red")
plt.title("loss(blue) vs validation loss(red)")
plt.show()

# Model Accuracy #

from keras.models import load_model
model=load_model('bestmodel.h5')
acc=model.evaluate(test_data)[1]
print(f"The Accuracy of Our Model by function is {acc*100}%")

# Own Accuracy Function #

from keras.preprocessing.image import load_img, img_to_array
Root_direc="test\LGG"
LGG_err_counter=0
LGG_loop_counter=0
#train_data.class_indices
for dir in os.listdir(Root_direc):
    LGG_loop_counter=LGG_loop_counter+1
    path ="test\LGG"+"/"+dir
    img=load_img(path,target_size=(240,240))
    input_arr=img_to_array(img)/255
    #input_arr.shape
    input_arr=np.expand_dims(input_arr,axis=0)
    pred= ((model.predict(input_arr)>0.5).astype("int32"))[0][0]
    if pred == 0:
        #print("Patient is having  HGG")
        LGG_err_counter=LGG_err_counter+1
    else:
        #print("Patient is  having LGG")
        continue

Root_direc="test\HGG"
HGG_err_counter=0
HGG_loop_counter=0
for dir in os.listdir(Root_direc):
    HGG_loop_counter=HGG_loop_counter+1
    path ="test\HGG"+"/"+dir
    img=load_img(path,target_size=(240,240))
    input_arr=img_to_array(img)/255
    #input_arr.shape
    input_arr=np.expand_dims(input_arr,axis=0)
    pred= ((model.predict(input_arr)>0.5).astype("int32"))[0][0]
    if pred == 0:
        #print("Patient is having  HGG")
        continue
    else:
        #print("Patient is  having LGG")
        HGG_err_counter=HGG_err_counter+1

# Accuracy Calculation

LGG_accuracy=1-(LGG_err_counter/LGG_loop_counter)
HGG_accuracy=1-(HGG_err_counter/HGG_loop_counter)
print(f"LGG Accuracy:{LGG_accuracy}%")
print(f"HGG Accuracy:{HGG_accuracy}%")
print(f"Model Accuracy(own function):{((LGG_accuracy+HGG_accuracy)/2)*100}%")

# Manual Testing Part #
path ="145\HGG\BraTS20_Training_226_t1ce_z78.png"
#path ="145\HGG\Y14.jpg"
#path ="145\LGG\BraTS20_Training_296_t1ce_z84.png"
#path ="test\HGG\BraTS20_Training_001_t1ce_z76.png"
img=load_img(path,target_size=(240,240))
input_arr=img_to_array(img)/255
plt.imshow(input_arr)
input_arr.shape
input_arr=np.expand_dims(input_arr,axis=0)
pred= ((model.predict(input_arr)>0.5).astype("int32"))[0][0]
#train_data.class_indices
if pred == 0:
    print("Patient is having  HGG")
else:
    print("Patient is  having LGG")

# Plotting #

temp=0
temp1=0
value=[]
validation=[]
for epoch in range(len(hs.epoch)):
    temp=temp+h['accuracy'][epoch]
    temp1=temp1+h['val_accuracy'][epoch]
    value.append(temp/(epoch+1))
    validation.append(temp1/(epoch+1))

plt.plot(value)
plt.plot(validation, c="red")
plt.title("Accuracy(blue) vs Validation Accuracy(red)")
plt.show()
#print(model.get_weights())
#print((model.predict(input_arr)>0.5))
#model.summary()
