# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import pandas as pd
import numpy as np
from keras.layers import Embedding, LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D, LeakyReLU
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from numpy import dstack
from keras import models
from keras import layers
import tensorflow as tf

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

scale_type = "MAX_ABS "


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# Carica una lista di files e restituisce un array 3D
# con dimensione è [samples, timesteps, features] (cioè [5000,315,3])

def load_group(filenames):
    loaded = list()
    for name in filenames:
        data = load_file(name)
        loaded.append(data)
    # feature in dimensione 3rd
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

#trasformazione formato files da csv a txt
def csvToTxt(path):
    text = open(path + ".csv", "r")
    text = ''.join([i for i in text]) \
        .replace(",", " ")
    x = open(path + ".txt", "w")
    x.writelines(text)
    x.close()

#caricamento dei dati
def loadData():
    csvToTxt('train_gesture_x')
    csvToTxt('train_gesture_y')
    csvToTxt('train_gesture_z')
    csvToTxt('train_label')
    # load data
    filenames = list()
    filenames += ['train_gesture_x.txt', 'train_gesture_y.txt', 'train_gesture_z.txt']
    trainX = load_group(filenames)
    trainY = load_file('train_label.txt')
    return trainX, trainY

def getNaCount2(filePath):
    dataset = pd.read_csv(filePath, header= None)
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente
    '''
    print("\n ----train", dataset.values)
    print("shape= ", dataset.shape)
    print("PROVA NAN = " ,dataset.values[0][314])
    '''

    df = pd.DataFrame(dataset.values)
    boolean_mask = df.isna()
    '''
    print("----mask: ", boolean_mask)
    print("SHAPE MASK ========= " , boolean_mask.shape)
    print(" PROVAAAA ========= " , boolean_mask[1])
    '''
    #la boolean mask viene scorsa per colonne
    for col in range(0,  boolean_mask.shape[1]) :   #colonne
        #print("col ===" , col)
        for row in range(0, boolean_mask.shape[0]): #righe
            #print("row ===", row)
            if boolean_mask[col][row] == True:
                #print("cecilia")
                #print("NAN == " , dataset.values[row][col])
                # se i = 0 --> prendi i+1
                # se i != 0 = 315 ???? --> prendi i-1
                if col == 0:
                    print(dataset.values[row][col+1])
                    substitutionValue = dataset.values[row][col+1]  #valore successivo nella riga j che metto al posto del nan
                    #dataset.values[row][col] = substitutionValue
                    dataset.at[row,col] = substitutionValue
                    #print("MERDA ==" , dataset.value[0][1])
                else :
                    #print("la colonna di merda e' -------", col)
                    #print(dataset.values[row][col - 1])
                    substitutionValue = dataset.values[row][col - 1]  # valore precedente nella riga j che metto al posto del nan
                    #dataset.values[row][col] = substitutionValue
                    #print("cacca ===",dataset.values[row][col] )
                    dataset.at[row, col] = substitutionValue

    dfAfterReplacement = pd.DataFrame(dataset.values)
    '''
    print("LIGLI CHE PALLE AIUTO ----------------------------------------------" , dataset.values[0][1])
    print("LIGLI CHE PALLE AIUTO ----------------------------------------------" , dataset.values[0][314])
    print("LIGLI CHE PALLE AIUTO ----------------------------------------------" , dataset.values[0][0])
    '''
    dfAfterReplacement.to_csv("provaNan.csv",  header= None, index = None)


# Expanding window
def kFoldValidation(trainX, trainY, testX, testY):
    print(" -------------  kFoldValidation   ---------")
    k = 10
    num_val_samples = len(trainX) // k  # 1000 per k=5

    print("len(trainX)", len(trainX))

    print("num_val_samples", num_val_samples)

    all_scores = []
    for i in range(k - 1):
        print('processing fold #', i)
        partial_trainX = trainX[: (i + 1) * num_val_samples]
        print("partial_trainX.shape", partial_trainX.shape)
        # print("partial_train_data", partial_train_data)

        partial_trainY = trainY[: (i + 1) * num_val_samples]
        print("partial_trainY.shape", partial_trainY.shape)
        # print("partial_train_targets", partial_train_targets)

        valX = trainX[(i + 1) * num_val_samples:(i + 1) * num_val_samples + num_val_samples]
        print("\n\nvalX.shape", valX.shape)
        # print("valX", val_data)
        valY = trainY[(i + 1) * num_val_samples:(i + 1) * num_val_samples + num_val_samples]
        print("valY.shape", valY.shape)
        # print("valY", val_targets)
        print("\n#########\n\n")

        # BUILD MODEL
        model = buildModel(partial_trainX, partial_trainY, valX, valY)
        #model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, validation_data=(val_data, val_targets),
        #          batch_size=64, verbose=1)
        testLoss, testAccuracy = model.evaluate(valX, valY, batch_size=128, verbose=1)
        all_scores.append(testAccuracy)

    print("\n\n ------- all_scores", all_scores)
    print("\n\n ------- np.mean(all_scores)", np.mean(all_scores))


def scale(datasetX, datasetY, scale_type):

    if scale_type == "STANDARD":
        scalers = {}
        for i in range(datasetX.shape[1]):
            scalers[i] = StandardScaler()
            datasetX[:, i, :] = scalers[i].fit_transform(datasetX[:, i, :])

        scalers2 = {}
        for i in range(datasetY.shape[1]):
            scalers2[i] = StandardScaler()
            datasetY[i:] = scalers2[i].fit_transform(datasetY[i:])


    if scale_type == "MINMAX":
        scalers = {}
        for i in range(datasetX.shape[1]):
            scalers[i] = MinMaxScaler(feature_range=(-1, 1))
            datasetX[:, i, :] = scalers[i].fit_transform(datasetX[:, i, :])

        scalers2 = {}
        for i in range(datasetY.shape[1]):
            scalers2[i] = MinMaxScaler(feature_range=(-1, 1))
            datasetY[i:] = scalers2[i].fit_transform(datasetY[i:])


    if scale_type == "MAX_ABS":
        scalers = {}
        for i in range(datasetX.shape[1]):
            scalers[i] =MaxAbsScaler()
            datasetX[:, i, :] = scalers[i].fit_transform(datasetX[:, i, :])

        scalers2 = {}
        for i in range(datasetY.shape[1]):
            scalers2[i] =MaxAbsScaler()
            datasetY[i:] = scalers2[i].fit_transform(datasetY[i:])

    if scale_type == "ROBUST":
        scalers = {}
        for i in range(datasetX.shape[1]):
            scalers[i] =RobustScaler(quantile_range=(25, 75), with_centering=False)
            datasetX[:, i, :] = scalers[i].fit_transform(datasetX[:, i, :])

        scalers2 = {}
        for i in range(datasetY.shape[1]):
            scalers2[i] = RobustScaler(quantile_range=(25, 75), with_centering=False)
            datasetY[i:] = scalers2[i].fit_transform(datasetY[i:])

    return datasetX, datasetY





def buildModel(datasetX, datasetY, testX, testY):
    verbose, epochs, batch_size = 1, 50, 64  # prima 64

    n_timesteps, n_features, n_outputs = datasetX.shape[1], datasetX.shape[2], datasetY.shape[1]
    print("n_timesteps == ", n_timesteps)  # 315
    print("n_features == ", n_features)  # 3
    print("n_outputs == ", n_outputs)  # 8


    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))

    model.add(MaxPooling1D(pool_size=4, strides=3, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(n_outputs, activation='softmax'))
    # model.compile(optimizer='adam', loss='mse')

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=1.e-3)
    opt2 = tf.keras.optimizers.RMSprop(lr=1e-4)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit network - training. ci stampa accuracy sui dati di training
    model.fit(datasetX, datasetY, epochs=epochs, batch_size=batch_size, verbose=verbose)

    '''
    # evaluate model. Ora controlliamo che il modello si comporti bene anche sul test set:
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=1)
    print("accuracy ==== ", accuracy)
    '''
    return model

#retrain del modello su 80% del training set completo
def finalTraining(trainX, trainY, testX, testY):
    model = buildModel(trainX, trainY, testX, testY)
    model.fit(trainX, trainY, epochs=50, batch_size=64, verbose=1)
    testLoss, testAccuracy = model.evaluate(testX, testY)

    print("\n\n ------- accuracy su TEST: ", testAccuracy)


def main():

    #caricamento dei files e conversione da csv in txt
    trainX, trainY = loadData()

    '''

    print("shape == ", trainX.shape)
    print(trainX)
    print(" \n=============================================\n")
    print("shape == ", trainY.shape)
    print(trainY)
    '''

    #scaling
    trainX, trainY = scale(trainX, trainY, scale_type)

    # split training/test (80/20)
    trainX, testX = train_test_split(trainX, test_size=.2, shuffle=False)
    trainY, testY = train_test_split(trainY, test_size=.2, shuffle=False)

    '''
    print("shape TRAIN == ", trainX.shape)
    print(trainX)
    print("\n\nshape TEST== ", testX.shape)
    print(testX)
    '''

    #print(" \n=============================================\n")


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

    #one-hot encoding
    trainY = tf.keras.utils.to_categorical(trainY, 8)
    testY = tf.keras.utils.to_categorical(testY, 8)

    #model = buildModel2(trainX, trainY, testX, testY)

    #cross validation
    kFoldValidation(trainX, trainY , testX, testY)

    #addestramento e valutazione del modello su training set(80%) e test set(20%)
    finalTraining(trainX, trainY , testX, testY)

    #todo RETRAIN FINALE SU TUTTO TRAINING!!
    #todo FUNZIONE PER TESTSET DI INPUT



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

