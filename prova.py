

import pandas as pd
import numpy as np
from keras.layers import Embedding, LSTM
from keras.utils.np_utils import to_categorical
from numpy import dstack
from keras import models
from keras import layers
import tensorflow as tf




def main():
    model = models.Sequential()
    model.add(layers.Dense(2, activation='relu', input_shape=(5,),
                           kernel_regularizer=tf.keras.regularizers.l2(1.e-4)))

    tf.keras.utils.plot_model(model)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()