import tensorflow as tf
import train


def main():

    # caricamento dei files e conversione da csv in txt
    filenameTrainX = "train_gesture_x"
    filenameTrainY = "train_gesture_y"
    filenameTrainZ = "train_gesture_z"
    filenameTrainLabel = "train_label"

    # Caricamento dei dati in formato 3D tensor
    trainX, trainY = train.loadData(filenameTrainX, filenameTrainY, filenameTrainZ, filenameTrainLabel)

    # controllo NaN nei files del test set
    #se ci sono nan, vengono opportunamente sostituiti e viene creato
    #un nuovo file csv con cui lavorare
    if train.naDetection("test_gesture_x") == True:
        filenameTestX = "test_gesture_x_withoutNaN"
    else:
        filenameTestX = "test_gesture_x"

    if train.naDetection("test_gesture_y") == True:
        filenameTestY = "test_gesture_y_withoutNaN"
    else:
        filenameTestY = "test_gesture_y"

    if train.naDetection("test_gesture_z") == True:
        filenameTestZ = "test_gesture_z_withoutNaN"
    else:
        filenameTestZ = "test_gesture_z"

    filenameTestLabel = "test_label"

    #Caricamento dei dati in formato 3D tensor
    testX, testY = train.loadData(filenameTestX, filenameTestY, filenameTestZ, filenameTestLabel)

    #scaling con max abs
    trainX, trainY , testX, testY= train.scale(trainX, trainY, testX, testY, "MAX_ABS")

    # one-hot encoding
    trainY = tf.keras.utils.to_categorical(trainY, 8)
    testY = tf.keras.utils.to_categorical(testY, 8)

    # valutazione accuracy sul test set
    train.finalTraining(trainX, trainY , testX, testY)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()