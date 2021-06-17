# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import pandas
from numpy import dstack

# load a single file as a numpy array
from pandas import read_csv


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
# We can then load all data for a given group (train or test) into a single three-dimensional NumPy array,
# where the dimensions of the array are [samples, time steps, features].
def load_group(filenames, folder):
    loaded = list()
    for name in filenames:
        data = load_file(folder + name)
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


def main2() :

    text = open("./data/train_gesture_x.csv", "r")
    text = ''.join([i for i in text]) \
        .replace(",", " ")
    x = open("output.txt", "w")
    x.writelines(text)
    x.close()

def main():
    # load data
    filenames = list()
    folder = 'data/'
    folder2 = 'Inertial Signals/'
    #filenames += ['train_gesture_x.txt', 'train_gesture_y.txt', 'train_gesture_z.txt']
    filenames += ['./data/train_gesture_x', './data/train_gesture_y', './data/train_gesture_z']
    filenames2 = replaceCommaCsv(filenames)

    ''' 
     filenames += ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
    # body acceleration
    filenames += ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt']
    # body gyroscope
    filenames += ['body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt']
    '''

    trainX = load_group(filenames2, folder)
    print("shape == " ,trainX.shape)

    print(trainX)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main2()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
