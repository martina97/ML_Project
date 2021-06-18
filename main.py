# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import pandas as pd
from numpy import dstack

# load a single file as a numpy array
from pandas import read_csv
from sklearn.preprocessing import StandardScaler


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

def replaceCommaCsv(filenames) :
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

    text = open(path+".csv", "r")
    text = ''.join([i for i in text]) \
        .replace(",", " ")
    x = open(path+".txt", "w")
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

    scaler = StandardScaler()
    x_train = scaler.fit_transform(dataset[0])
    print("x_train: ", x_train)


    for i in range(0,dataset.shape[0]):

        print ("i= ", i)

        bho = pd.DataFrame(dataset[i])
        #print("\n---bho[0] ", bho)
        boolean_mask = bho.isna()
        # contiamo il numero di TRUE per ogni attributo sul dataset
        '''
                if (boolean_mask == False):
            print("MERDA")
        
        '''


        print("----mask: ", boolean_mask)
        count = boolean_mask.sum(axis=0)

        print("count NaN: ",count)
        print("count iloc sto cazzo: ",count.values)

        stronzo = 1 in count.values
        print("stronzo = ", stronzo)





def main():

    trainX, trainY = loadData()

    print("shape == ", trainX.shape)
    print(trainX)
    print(" \n=============================================\n")
    print("shape == ", trainY.shape)
    print(trainY)

    print(getNaCount(trainX))

    #todo ora fare standard scaler


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
