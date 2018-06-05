import numpy as np
import scipy as sps
import pandas as pd


from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from rnn_test.utils.keras_utils import train_model, evaluate_model, set_trainable
from rnn_test.utils.layer_utils import AttentionLSTM

DATASET_INDEX = 14

MAX_TIMESTEPS = 1609
MAX_NB_VARIABLES = 42
NB_CLASS = 2

TRAINABLE = True


def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    x = Masking()(ip) # if all values in the input tensor at that timestep are equal to mask_value, then the timestep will be masked
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

batch_size = 64

for i in range (5,10):
    model = generate_model()
    train_model(model, DATASET_INDEX, dataset_prefix='ozone', dataset_fold_id = i,epochs=300, batch_size=batch_size)
    evaluate_model(model, DATASET_INDEX, dataset_prefix='ozone',dataset_fold_id = i, batch_size=batch_size, iteration = i)

'''
arr = (np.load("dataset/x_test.npy")) # (173, 72, 291) -> (173, 1)
print(arr.shape)
'''