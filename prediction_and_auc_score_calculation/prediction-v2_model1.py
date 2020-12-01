#!/usr/bin/env python
# coding: utf-8

# In[2]:

from IPython.display import Image, display

get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow-gpu==2.2')


# In[3]:


get_ipython().system('pip install -U scikit-learn')


# In[4]:


get_ipython().system('pip install pandas')


# In[5]:


get_ipython().system('pip install h5py')


# In[6]:


get_ipython().system('pip install matplotlib')


# In[7]:


get_ipython().system('pip install pillow')


# In[8]:


get_ipython().system('pip install scikit-plot')


# In[8]:


# Importing the Keras libraries and packages
import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout, Concatenate
from keras.layers import AveragePooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Cropping2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

from keras.layers.core import Lambda, Flatten, Dense


from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.preprocessing import image

import datetime

from keras.utils import multi_gpu_model

from keras.preprocessing.image import ImageDataGenerator

from os import listdir
import os, random, shutil
from os.path import isfile, join
from IPython.display import Image, display , clear_output
import time

import pandas as pd

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model

# from IPython.display import Image, display , clear_output


# In[9]:


import sys
import numpy as np
import pandas as pd
# from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt

import keras

import time
from keras.optimizers import SGD

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate,Conv1D
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng

from keras.layers import Dropout, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Cropping2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, GlobalMaxPooling1D


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[11]:


batch_size=32


train_datagen = ImageDataGenerator(
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   vertical_flip = True,
                                   horizontal_flip = True)


test_datagen = ImageDataGenerator(                                 )

training_set = train_datagen.flow_from_directory('data/Florida/day1_full/train/',
                                                 target_size = (400, 225),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('data/Florida/test_3096/',
                                            target_size = (400, 225),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
test_set2 = test_datagen.flow_from_directory('data/Florida/test_3096/',
                                            target_size = (400, 225),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

img_height = 225
img_width = 400
dir1= 'data/Florida/day1_full/train/'
# dir2= 'data/Florida/test_1/'
dir2= 'data/Florida/test_3096/'


input_imgen = ImageDataGenerator(#rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2,
                                   rotation_range=5.,
                                   horizontal_flip = True)

test_imgen = ImageDataGenerator(#rescale = 1./255
                               )


def generate_generator_multiple(generator,dir1, dir2, batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          class_mode = 'binary',
                                          batch_size = batch_size
                                         )
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          class_mode = 'binary',
                                          batch_size = batch_size
                                         )
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
            
            
inputgenerator=generate_generator_multiple(generator=input_imgen,
                                           dir1=dir1,
                                           dir2=dir1,
                                           batch_size=batch_size,
                                           img_height=img_height,
                                           img_width=img_width)       
     
testgenerator=generate_generator_multiple(test_imgen,
                                          dir1=dir2,
                                          dir2=dir2,
                                          batch_size=batch_size,
                                          img_height=img_height,
                                          img_width=img_width)  

     
testgenerator1=generate_generator_multiple(test_imgen,
                                          dir1=dir2,
                                          dir2=dir2,
                                          batch_size=batch_size,
                                          img_height=img_height,
                                          img_width=img_width)    

steps_per_epoch =round(len(training_set))
validation_steps =round(len(test_set))
print(steps_per_epoch)
print(validation_steps)

# print(test_set.class_indices)  


# In[12]:


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
# #################################
    input_shape=(400, 225, 3)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Initialising the CNN    
    # Convolutional Neural Network
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Adding a third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Adding a fourth convolutional layer
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    model.add(GlobalMaxPooling2D())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
#     keras.utils.plot_model(siamese_net, "multi_input_and_output_model.png", show_shapes=True)
    # return the model
    return siamese_net


model = get_siamese_model((400, 225, 3))
model.summary()



#Model 1
model.load_weights("weights_imported/siamese_4layers_day1_v1-epoch34.h5")


from pathlib import Path

test_samples = []
test_labels = []

list1 = (os.listdir("data/Florida/test_set/change/")) 
list2 = (os.listdir("data/Florida/test_set/same/")) 

print(len(list1))
print(len(list2))
count=0
j=0
# while j<675:
while j<64810:

    testnewset1 = (os.listdir("data/Florida/test_set/all_frames/")) 

    testnewset_sorted1 = sorted(testnewset1) 
    res1 = testnewset_sorted1[j] 

    res1 =Path(f'data/Florida/test_set/all_frames/{res1}').stem
    


    test_img_path2= f'data/Florida/test_set/change/{res1}.jpg'
    
    test_samples.append(res1)

    if ( not os.path.isfile(test_img_path2)):
        #same
        label =[0]
        test_labels.append(label)
    else:
#       #change
        label =[1]
        test_labels.append(label)
    
    os.system('cls||clear')
    print(count)
    count+=1

    j+=1

test_labels = (np.array(test_labels))

list = pd.DataFrame(np.column_stack([test_samples,test_labels ]), 
                               columns=['Sample', 'True labels'])
print(list)

from os import listdir
import os, random, shutil
from os.path import isfile, join
import time
from pathlib import Path

import numpy as np
from random import shuffle

global k

def file():
        j =0

        while j <1:

            testnewset1 = (os.listdir("data/Florida/test_set/all_frames/")) 

            testnewset_sorted1 = sorted(testnewset1) 

            res1 = testnewset_sorted1[k] 
            res2 = testnewset_sorted1[k+4] ##########

            print ("The frame number is : " +  str(res1)) 
            print ("The frame number is : " +  str(res2)) 


            test_img_path1 = f"data/Florida/test_set/all_frames/{res1}" 
            test_img_path2 = f"data/Florida/test_set/all_frames/{res2}" 


            test_image1 = image.load_img(test_img_path1, target_size = (400, 225))
            test_image1 = image.img_to_array(test_image1)
            test_image1 = np.expand_dims(test_image1, axis = 0)

            test_image2 = image.load_img(test_img_path2, target_size = (400, 225))
            test_image2 = image.img_to_array(test_image2)
            test_image2 = np.expand_dims(test_image2, axis = 0)

            j+=1

        return test_image1,test_image2

    
test_all_frames=[]
test_pred_labels=[]
test_pred_labels_with_threshold=[]

k=0
while k<64810:

    result = model.predict(file(),batch_size=1, verbose=1)
#     os.system('cls||clear')

    print(result)
    preds = [float(result)]

    test_pred_labels_with_threshold.append(preds)
    list3 = (os.listdir("data/Florida/test_set/all_frames/"))     

    list3 = sorted(list3)
    res3 = list3[k] 
    test_all_frames.append(res3)
    k+=1
    clear_output(wait=True)


test_pred_labels_with_threshold = (np.array(test_pred_labels_with_threshold))
print(len(test_pred_labels_with_threshold))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

test_response = test_labels
scores = test_pred_labels_with_threshold

fpr, tpr, thresholds = roc_curve(test_response, scores)
roc_auc = roc_auc_score(test_response, scores)
print("AUC of ROC Curve:", roc_auc)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig('roc_curve_model1_t+4.png')
