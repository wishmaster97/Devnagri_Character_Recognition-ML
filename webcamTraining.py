import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils import np_utils, print_summary
from keras import optimizers
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K

data_train = pd.read_csv("devnagridata.csv")
data_test = pd.read_csv("datatest.csv")
dataset_train = np.array(data_train)
dataset_test = np.array(data_test)
np.random.shuffle(dataset_train)
np.random.shuffle(dataset_test)

X = dataset_train
Y = dataset_train
X = X[:, 0:1024]
Y = Y[:, 1024]

X_t = dataset_test
Y_t = dataset_test
X_t = X_t[:, 0:1024]
Y_t = Y_t[:, 1024]

X_train = X[0:72001, :]
X_train = X_train / 255.
X_test = X_t[0:21601, :]
X_test = X_test/ 255.

#Reshape
Y = Y.reshape(Y.shape[0], 1)
Y_t = Y_t.reshape(Y_t.shape[0], 1)
Y_train = Y[0:72001, :]
Y_train = Y_train.T
Y_test  = Y_t[0:21601, :]
Y_test  = Y_test.T

print("Number of Training examples = "+str(X_train.shape[0]))
print("Number of test examples = "+str(X_test.shape[0]))
print("X_train shape: "+str(X_train.shape))
print("Y_train shape: "+str(Y_train.shape))
print("X_test shape: "+str(X_test.shape))
print("Y_test shape: "+str(Y_test.shape))

image_x = 32
image_y = 32

train_y = np_utils.to_categorical(Y_train)
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y  = np_utils.to_categorical(Y_test)
test_y  = test_y.reshape(test_y.shape[1], test_y.shape[2])
X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
X_test  = X_test.reshape(X_test.shape[0], image_x, image_y, 1)

print("X_train shape: "+str(X_train.shape))
print("Y_train shape: "+str(train_y.shape))

#Buildinig A Model

def keras_model(image_x,image_y):
    num_of_classes = 37
    model = Sequential()
    model.add(Conv2D(filters =32, kernel_size =(5,5), input_shape = (image_x, image_y, 1), activation ='relu'))
    model.add(MaxPooling2D(pool_size =(2,2), strides=(2,2), padding='same'))
    model.add(Conv2D(64,(5,5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (5,5), strides=(5,5), padding ='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation = 'softmax'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics =['accuracy'])
    filepath = "devnagri_model_refined.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode= 'max')
    callbacks_list = [checkpoint1]
    return model,callbacks_list

model, callbacks_list = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data =(X_test,test_y),epochs =20, batch_size = 64, callbacks = callbacks_list)
scores = model.evaluate(X_test, test_y, verbose =0)
print("CNN Error : %.2f%%" % (100-scores[1]*100))
print_summary(model)
model.save('devnagri.h5')


#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
