
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

scale_type = "MAX_ABS"
epochs = 50
batch_size = 64


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# Carica una lista di files e restituisce un array 3D
# con dimensione 3 [samples, timesteps, features] (cioè [5000,315,3])
def load_group(filenames):
    loaded = list()
    for name in filenames:
        data = load_file(name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


#trasformazione formato files da csv a txt
def csvToTxt(path):
    text = open(path + ".csv", "r")
    text = ''.join([i for i in text]) \
        .replace(",", " ")
    x = open(path + ".txt", "w")
    x.writelines(text)
    x.close()


#caricamento dei dati
def loadData(pathX, pathY, pathZ, pathLabel):
    csvToTxt(pathX)
    csvToTxt(pathY)
    csvToTxt(pathZ)
    csvToTxt(pathLabel)
    # load data
    filenames = list()
    filenames += [pathX+".txt", pathY+".txt", pathZ+".txt"]
    datasetX = load_group(filenames)
    datasetY = load_file(pathLabel+".txt")
    return datasetX, datasetY

 # Sostituiamo i valori mancanti (NaN) del train e del test con opportuni valori e, in caso di NaN nel dataset,
 #si crea un nuovo csv
def naDetection(filePath):
    """
    Ritorna True se nel dataset sono stati trovati NaN, altrimenti False.
    """

    dataset = pd.read_csv(filePath +".csv", header= None)
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente

    df = pd.DataFrame(dataset.values)
    boolean_mask = df.isna()
    #controllo se ci sono Nan
    countNan = boolean_mask.sum(axis=0)
    checkIfNan = 1 in countNan.values
    print("checkIfNan = ", checkIfNan)

    if checkIfNan :

        # la boolean mask viene scorsa per colonne
        for col in range(0, boolean_mask.shape[1]):  # colonne
            # print("col ===" , col)
            for row in range(0, boolean_mask.shape[0]):  # righe
                # print("row ===", row)
                if boolean_mask[col][row] == True:  #NaN

                    # se i = 0 --> sostituisci la cella contenente il NaN con il valore nella cella successiva (i+1)
                    # altrimenti  --> sostituisci la cella contenente il NaN con il valore nella cella successiva (i-1)
                    if col == 0:
                        print(dataset.values[row][col + 1])
                        substitutionValue = dataset.values[row][col + 1]  # valore successivo nella riga j che metto al posto del nan
                        dataset.at[row, col] = substitutionValue     #sostituzione NaN

                    else:
                        substitutionValue = dataset.values[row][col - 1]  # valore precedente nella riga j che metto al posto del nan
                        dataset.at[row, col] = substitutionValue           #sostituzione NaN

        # nuovo dataFrame con i NaN sostituiti che viene scritto in un nuovo csv
        dfAfterReplacement = pd.DataFrame(dataset.values)
        dfAfterReplacement.to_csv(filePath + "_withoutNaN.csv", header=None, index=None)
        return checkIfNan
    else:
        #false (no NaN trovati)
        return checkIfNan


# Expanding window cross validation
def kFoldValidation(trainX, trainY):

    k = 10
    num_val_samples = len(trainX) // k  #cardinalita' validation set

    '''
    print("len(trainX)", len(trainX))
    print("num_val_samples", num_val_samples)
    '''

    all_scores = []
    for i in range(k - 1):
        partial_trainX = trainX[: (i + 1) * num_val_samples]
        #print("partial_trainX.shape", partial_trainX.shape)

        partial_trainY = trainY[: (i + 1) * num_val_samples]
        #print("partial_trainY.shape", partial_trainY.shape)

        valX = trainX[(i + 1) * num_val_samples:(i + 1) * num_val_samples + num_val_samples]
        #print("\n\nvalX.shape", valX.shape)

        valY = trainY[(i + 1) * num_val_samples:(i + 1) * num_val_samples + num_val_samples]
        #print("valY.shape", valY.shape)


        # BUILD MODEL
        model = buildModel(partial_trainX, partial_trainY)
        model.fit(partial_trainX, partial_trainY, epochs=50, batch_size=64, verbose=1)

        testLoss, testAccuracy = model.evaluate(valX, valY, batch_size=128, verbose=1)
        all_scores.append(testAccuracy)

    print("\n\n ------- all_scores", all_scores)
    print("\n\n ------- np.mean(all_scores)", np.mean(all_scores))


def scale(trainX, trainY, testX, testY, scale_type):
    if scale_type == "STANDARD":
        scaler = StandardScaler()
        # datasetX = scaler.fit_transform(datasetX)
        scalers = {}
        for i in range(trainX.shape[1]):
            scalers[i] = StandardScaler()
            trainX[:, i, :] = scalers[i].fit_transform(trainX[:, i, :])
        scalers2 = {}
        for i in range(trainY.shape[1]):
            scalers2[i] = StandardScaler()
            trainY[i:] = scalers2[i].fit_transform(trainY[i:])
        if testX is not None:
            for i in range(testX.shape[1]):
                testX[:, i, :] = scalers[i].transform(testX[:, i, :])
            for i in range(testY.shape[1]):
                testY[i:] = scalers2[i].transform(testY[i:])
        print("DIM TRAIN DOPO SCALER ", trainX.shape)
    if scale_type == "MINMAX":
        # ToDo cambiare range
        scalers = {}
        for i in range(trainX.shape[1]):
            scalers[i] = MinMaxScaler(feature_range=(-1, 1))
            trainX[:, i, :] = scalers[i].fit_transform(trainX[:, i, :])
        scalers2 = {}
        for i in range(trainY.shape[1]):
            scalers2[i] = MinMaxScaler(feature_range=(-1, 1))
            trainY[i:] = scalers2[i].fit_transform(trainY[i:])
        if testX is not None:
            for i in range(testX.shape[1]):
                testX[:, i, :] = scalers[i].transform(testX[:, i, :])
            for i in range(testY.shape[1]):
                testY[i:] = scalers2[i].transform(testY[i:])

    if scale_type == "MAX_ABS":
        scalers = {}
        for i in range(trainX.shape[1]):
            scalers[i] =MaxAbsScaler()
            trainX[:, i, :] = scalers[i].fit_transform(trainX[:, i, :])
        scalers2 = {}
        for i in range(trainY.shape[1]):
            scalers2[i] =MaxAbsScaler()
            trainY[i:] = scalers2[i].fit_transform(trainY[i:])
        if testX is not None:
            for i in range(testX.shape[1]):
                testX[:, i, :] = scalers[i].transform(testX[:, i, :])
            for i in range(testY.shape[1]):
                testY[i:] = scalers2[i].transform(testY[i:])

    if scale_type == "ROBUST":
        scalers = {}
        for i in range(trainX.shape[1]):
            scalers[i] =RobustScaler(quantile_range=(25, 75), with_centering=False)
            trainX[:, i, :] = scalers[i].fit_transform(trainX[:, i, :])
        scalers2 = {}
        for i in range(trainY.shape[1]):
            scalers2[i] = RobustScaler(quantile_range=(25, 75), with_centering=False)
            trainY[i:] = scalers2[i].fit_transform(trainY[i:])
        if testX is not None:
            for i in range(testX.shape[1]):
                testX[:, i, :] = scalers[i].transform(testX[:, i, :])
            for i in range(testY.shape[1]):
                testY[i:] = scalers2[i].transform(testY[i:])
    return trainX, trainY, testX, testY



def buildModel(datasetX, datasetY):

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

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=1.e-3)
    opt2 = tf.keras.optimizers.RMSprop(lr=1e-4)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def finalTraining(trainX, trainY, testX, testY):
    model = buildModel(trainX, trainY)
    # fit network - training. ci stampa accuracy sui dati di training
    model.fit(trainX, trainY, epochs=50, batch_size=23, verbose=1)
    testLoss, testAccuracy = model.evaluate(testX, testY)

    print("\n\n ------- accuracy su TEST: ", testAccuracy)


def main():


    filenameTrainX = "train_gesture_x"
    filenameTrainY = "train_gesture_y"
    filenameTrainZ = "train_gesture_z"
    filenameTrainLabel = "train_label"

    #Caricamento dei dati in formato 3D tensor
    trainX, trainY = loadData(filenameTrainX, filenameTrainY, filenameTrainZ, filenameTrainLabel)

    # split training / test --> 80/20
    trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=.2, shuffle=False)

    # scaling con scaler di tipo "scale_type"
    trainX, trainY, testX, testY, = scale(trainX, trainY, testX, testY, scale_type)
    #todo scale dopo one-hot enc


    #one-hot encoding
    trainY = tf.keras.utils.to_categorical(trainY, 8)
    testY = tf.keras.utils.to_categorical(testY, 8)


    #extended window cross validation
    kFoldValidation(trainX, trainY)

    #valutazione accuracy sul test set
    finalTraining(trainX, trainY , testX, testY)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()