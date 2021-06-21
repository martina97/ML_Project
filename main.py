# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import pandas as pd
import numpy as np
from keras.layers import Embedding, LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from numpy import dstack
from keras import models
from keras import layers
import tensorflow as tf

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

scale_type = "STANDARD"

def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
# We can then load all data for a given group (train or test) into a single three-dimensional NumPy array,
# where the dimensions of the array are [samples, time steps, features].
def load_group(filenames):
    loaded = list()
    for name in filenames:
        data = load_file(name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


def replaceCommaCsv(filenames):
    filenames2 = list()
    for name in filenames:
        data = ""

        with open(name + '.csv') as file:
            data = file.read().replace(",", " ")

        with open("output2.txt") as file:
            file.write(data)

        filenames2 += [name + '.txt']
    return filenames2


def csvToTxt(path):
    text = open(path + ".csv", "r")
    text = ''.join([i for i in text]) \
        .replace(",", " ")
    x = open(path + ".txt", "w")
    x.writelines(text)
    x.close()


def loadData():
    # load data
    filenames = list()
    filenames += ['train_gesture_x.txt', 'train_gesture_y.txt', 'train_gesture_z.txt']
    trainX = load_group(filenames)

    csvToTxt('train_label')
    trainY = load_file('train_label.txt')

    return trainX, trainY


def getNaCount(dataset):
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente

    print("\n ----train[0] ", dataset[0])
    print("shape= ", dataset.shape[0])


    #print("x_train: ", x_train)

    for i in range(0, dataset.shape[0]):
        print("i= ", i)

        bho = pd.DataFrame(dataset[i])
        # print("\n---bho[0] ", bho)
        boolean_mask = bho.isna()
        # contiamo il numero di TRUE per ogni attributo sul dataset
        '''
                if (boolean_mask == False):
            print("MERDA")

        '''

        print("----mask: ", boolean_mask)
        count = boolean_mask.sum(axis=0)

        print("count NaN: ", count)
        print("count iloc sto cazzo: ", count.values)

        stronzo = 1 in count.values
        print("stronzo = ", stronzo)

# Expanding window
def kFoldValidation(trainX, trainY):

    print(" -------------  kFoldValidation   ---------")
    k = 10
    num_val_samples = len(trainX) // k  #1000 per k=5 --> FISSO!

    print("len(trainX)" ,  len(trainX))

    print("num_val_samples" , num_val_samples)

    num_epochs = 100
    all_scores = []
    for i in range(k-1):
        print('processing fold #', i)
        partial_train_data = trainX[: (i + 1) * num_val_samples]
        #val_data = trainX[i * num_val_samples: (i + 1) * num_val_samples]
        print("partial_train_data.shape", partial_train_data.shape)
        #print("partial_train_data", partial_train_data)

        partial_train_targets = trainY[: (i + 1) * num_val_samples]
        print("partial_train_targets.shape", partial_train_targets.shape)
        #print("partial_train_targets", partial_train_targets)

        val_data = trainX[(i + 1) * num_val_samples:(i + 1) * num_val_samples + num_val_samples]
        print("\n\nval_data.shape", val_data.shape)
        #print("val_data", val_data)
        val_targets = trainY[(i + 1) * num_val_samples:(i + 1) * num_val_samples + num_val_samples]
        print("val_targets.shape", val_targets.shape)
        #print("val_targets", val_targets)
        print("\n#########\n\n")

        #BUILD MODEL
        model = buildModel(partial_train_data)
        model.fit(partial_train_data, partial_train_targets,epochs=num_epochs,validation_data=(val_data, val_targets),  batch_size=64, verbose=1)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

    print("\n\n ------- all_scores", all_scores)
    print("\n\n ------- np.mean(all_scores)", np.mean(all_scores))


def scale(datasetX, datasetY, scale_type) :

    if scale_type == "STANDARD":

        scaler = StandardScaler()

        #datasetX = scaler.fit_transform(datasetX)
        scalers = {}
        for i in range(datasetX.shape[1]):
            scalers[i] = StandardScaler()
            datasetX[:, i, :] = scalers[i].fit_transform(datasetX[:, i, :])
        print("DIM TRAIN DOPO SCALER " ,datasetX.shape)

        ''' 
        datasetY = scaler.fit_transform(datasetY)
        
        datasetX = scaler.fit_transform(datasetX.reshape(-1, datasetX.shape[-1])).reshape(datasetX.shape)
        datasetY = scaler.transform(datasetY.reshape(-1, datasetY.shape[-1])).reshape(datasetY.shape)

        scalers = {}
        for i in range(datasetX.shape[1]):
            scalers[i] = StandardScaler()
            datasetX[:, i, :] = scalers[i].fit_transform(datasetX[:, i, :])

        '''

    if scale_type == "MINMAX":
        #ToDo cambiare range
        scaler = MinMaxScaler(feature_range=(-1, 1))
        datasetX = scaler.fit_transform(datasetX[0])

    if scale_type == "MAX_ABS":
        scaler = MaxAbsScaler()
        datasetX = scaler.fit_transform(datasetX[0])

    if scale_type == "ROBUST":
        #ToDo cambiare range
        scaler =RobustScaler(quantile_range=(25, 75), with_centering=False)
        datasetX = scaler.fit_transform(datasetX[0])

    return datasetX,   datasetY


def buildModel(trainX):
    model = models.Sequential()
    #model.add(layers.LSTM(32))
    model.add(layers.Dense(315, activation='relu', input_shape=(trainX.shape[1],), kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))

    '''
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='relu'))  #PRIMA ERA 64
    model.add(layers.Dense(9, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    '''
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(.3))

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(9, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1.e-4)))

    opt = tf.keras.optimizers.Adam(learning_rate=1.e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def buildModel2(datasetX, datasetY, testX, testY) :
    verbose, epochs, batch_size = 1, 23, 128 #prima 64

    n_timesteps, n_features, n_outputs = datasetX.shape[1], datasetX.shape[2], datasetY.shape[1]
    print("n_timesteps == " , n_timesteps) # 315
    print("n_features == " , n_features)    # 3
    print("n_outputs == " , n_outputs)  # 8
    model = Sequential()
    #model.add(layers.Dense(200, activation='relu', input_shape=(n_timesteps,), kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))

    '''
    model.add(LSTM(315, input_shape=(n_timesteps, n_features)))
    #model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    '''

    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))

    model.add(MaxPooling1D(pool_size=4,  strides=3, padding='valid'))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))


    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(n_outputs, activation='softmax'))
    #model.compile(optimizer='adam', loss='mse')

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(datasetX, datasetY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=1)
    print("accuracy ==== ", accuracy)

def main():
    trainX, trainY = loadData()
    '''

    print("shape == ", trainX.shape)
    print(trainX)
    print(" \n=============================================\n")
    print("shape == ", trainY.shape)
    print(trainY)
    '''
    #print("PROVA SIMOOOO " , trainX[0].shape)

    trainX, trainY = scale(trainX, trainY, scale_type)



    #split training / test --> 80/20
    trainX, testX = train_test_split(trainX, test_size=.2, shuffle=False)



    '''
    print("shape TRAIN == ", trainX.shape)
    print(trainX)
    print("\n\nshape TEST== ", testX.shape)
    print(testX)
    '''

    print(" \n=============================================\n")

    trainY, testY = train_test_split(trainY, test_size=.2, shuffle=False)
    ''' 
    print("shape TRAIN == ", trainY.shape)
    print(trainY)
    print("\n\nshape TEST== ", testY.shape)
    print(testY)
    '''

    ''' 
    trainY = to_categorical(trainY, 3)
    testY = to_categorical(testY, 3)
    
    '''

    trainY = tf.keras.utils.to_categorical(trainY,8)
    testY = tf.keras.utils.to_categorical(testY,8)

    #model = buildModel(trainX)

    model = buildModel2(trainX, trainY, testX, testY)

    #kFoldValidation(trainX, trainY)
    #print(getNaCount(trainX))



    # todo ora fare ONE-HOT ENCODING

    '''
    1 split train test
    2 prep (na, outlier (?), scaling -> standardizzare i dati, one hot)
    3 validation (regolarizzaz si fa dentro all'add dei layers)
    4 model fit 
    5 data agumentation


    '''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/