# -*- coding: utf-8 -*-
"""vggish_with_augmented_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zzzthBSoSRNlybGg0TxhysbYLS4DmpOo
"""

# !pip list
from __future__ import print_function
from __future__ import absolute_import

# from google.colab import drive
# drive.mount('/content/drive')

# !zip -r data.zip /content/data/*

# !cat vggish_params.py

# !unzip ./data.zip

# !git clone https://github.com/DTaoo/VGGish.git
# !cp VGGish/* .
# !pip install numpy resampy tensorflow tf_slim six soundfile

# !pip install -q -U keras-tuner

# !unzip '/content/drive/MyDrive/PTSD dataset/saved_ds.zip'

import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
import datetime
from posixpath import dirname
import pickle
# import keras
# from preprocess_sound import preprocess_sound
# import keras_tuner as kt
import matplotlib.pyplot as plt

import mel_features
import vggish_params as params
import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D
# from keras.engine.topology import get_source_inputs
from tensorflow.keras import backend as K

# Params
# path = '/content/drive/MyDrive/PTSD dataset' # Base data path # Should contain folders yes_ptsd and no_ptsd
path = './content/data'

# Saved dataset after preprocessing
# SAVED_DATASET_PATH = '/content/saved_ds'
SAVED_DATASET_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/saved_ds_clean2_balanced_30s"
SAVED_MAIN_DATASET_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/saved_ds_clean2_balanced_30s"
SAVED_DATASET_PATH_2 = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/saved_ds_augm1_clean2_balanced_30s"
SAVED_DATASET_PATH_3 = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/saved_ds_augm2_clean2_balanced_30s"
SAVED_DATASET_PATH_4 = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/saved_ds_augm3_clean2_balanced_30s"

TRAIN_DS_PATH = os.path.join(SAVED_MAIN_DATASET_PATH, 'train_ds')
TRAIN_DS_PATH_2 = os.path.join(SAVED_DATASET_PATH_2, 'train_ds')
TRAIN_DS_PATH_3 = os.path.join(SAVED_DATASET_PATH_3, 'train_ds')
TRAIN_DS_PATH_4 = os.path.join(SAVED_DATASET_PATH_4, 'train_ds')

TEST_DS_PATH = os.path.join(SAVED_MAIN_DATASET_PATH, 'test_ds')

VAL_DS_PATH = os.path.join(SAVED_MAIN_DATASET_PATH, 'val_ds')

USE_FOURTH_DS = False

# VGGish model weigths paths
WEIGHTS_PATH = '/raid/home/labusermoctar/ptsd_audio_detection/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = '/raid/home/labusermoctar/ptsd_audio_detection/vggish_audioset_weights.h5'

# Checkpoints
CHECKPOINT_FILEPATH = './final_training/checkpoint'
SAVED_MODELS_PATH = "./final_training/saved_models"

# Epochs
EPOCHS = 150

# FLAG
IS_COLAB = False

SEED = 42

AUTO_BALANCE = True

classes = ["Without PTSD", "With PTSD"]

EXAMPLE_WINDOW_SECONDS = 29.76
EXAMPLE_HOP_SECONDS = 29.76
NUM_FRAMES = 2976

BATCH_SIZE = 128

USE_SAVED_DATASET = True

INPUT_SHAPE = (NUM_FRAMES, params.NUM_BANDS, 1)
# INPUT_SHAPE = (2976, 64, 1)

tf.random.set_seed(SEED)

train_ds = tf.data.experimental.load(TRAIN_DS_PATH).unbatch().batch(BATCH_SIZE) \
    .concatenate(tf.data.experimental.load(TRAIN_DS_PATH_2).unbatch().batch(BATCH_SIZE)) \
    .concatenate(tf.data.experimental.load(TRAIN_DS_PATH_3).unbatch().batch(BATCH_SIZE))
# if USE_FOURTH_DS:
#     train_ds = train_ds.concatenate(tf.data.experimental.load(TRAIN_DS_PATH_4).unbatch().batch(BATCH_SIZE))

test_ds = tf.data.experimental.load(TEST_DS_PATH).unbatch().batch(BATCH_SIZE)  # \


val_ds = tf.data.experimental.load(VAL_DS_PATH).unbatch().batch(BATCH_SIZE)

print(len(list(train_ds.unbatch().as_numpy_iterator())))

X_train = np.concatenate([x for x, y in train_ds], axis=0)
print(X_train.shape)
y_train = np.concatenate([y for x, y in train_ds], axis=0)
y_train = tf.keras.utils.to_categorical(y_train)
print(y_train.shape)

X_val = np.concatenate([x for x, y in val_ds], axis=0)
print(X_val.shape)
y_val = np.concatenate([y for x, y in val_ds], axis=0)
y_val = tf.keras.utils.to_categorical(y_val)
print(y_val.shape)

X_test = np.concatenate([x for x, y in test_ds], axis=0)
print(X_test.shape)
y_test = np.concatenate([y for x, y in test_ds], axis=0)
y_test = tf.keras.utils.to_categorical(y_test)
print(y_test.shape)


def plot_acc(history, end_offset=0):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    if end_offset:
        acc = acc[:-end_offset]
        val_acc = val_acc[:-end_offset]
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"final_training/model_acc_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png")
    plt.show()


def plot_loss(history, end_offset=0):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    if end_offset:
        loss = loss[:-end_offset]
        val_loss = val_loss[:-end_offset]
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"final_training/model_loss_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png")
    plt.show()


def get_source_inputs(*args, **kwargs):
    ...


"""Loading and building model"""

"""VGGish model for Keras. A VGG-like model for audio classification
# Reference
- [CNN Architectures for Large-Scale Audio Classification](ICASSP 2017)
"""

def VGGish(load_weights=True, weights='audioset',
           input_tensor=None, input_shape=None,
           out_dim=None, include_top=True, pooling='avg'):
    '''
    An implementation of the VGGish architecture.
    :param load_weights: if load weights
    :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
    :param input_tensor: input_layer
    :param input_shape: input data shape
    :param out_dim: output dimension
    :param include_top:whether to include the 3 fully-connected layers at the top of the network.
    :param pooling: pooling type over the non-top network, 'avg' or 'max'
    :return: A Keras model instance.
    '''

    if weights not in {'audioset', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `audioset` '
                         '(pre-training on audioset).')

    if out_dim is None:
        out_dim = params.EMBEDDING_SIZE

    # input shape
    if input_shape is None:
        # input_shape = (params.NUM_FRAMES, params.NUM_BANDS, 1)
        input_shape = (NUM_FRAMES, params.NUM_BANDS, 1)

    if input_tensor is None:
        aud_input = Input(shape=input_shape, name='input_1')
    else:
        if not K.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

    if include_top:
        # FC block
        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
        x = Dense(out_dim, activation="relu", name='vggish_fc2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    # Create model.
    model = Model(inputs, x, name='VGGish')

    # load weights
    if load_weights:
        if weights == 'audioset':
            if include_top:
                model.load_weights(WEIGHTS_PATH_TOP)
            else:
                model.load_weights(WEIGHTS_PATH)
        else:
            print("failed to load weights")

    return model


base_model_ = VGGish(load_weights=False, include_top=False, input_shape=INPUT_SHAPE, pooling='avg')
base_model_.load_weights(WEIGHTS_PATH, by_name=True)
# base_model_.trainable = False
base_model_.summary()
# base_model_.layers.pop()

model_ = tf.keras.models.Sequential()
model_.add(base_model_)
# model_.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_constraint=tf.keras.constraints.MaxNorm(0.1), kernel_regularizer=tf.keras.regularizers.L2(0.02)))
model_.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                 kernel_regularizer=tf.keras.regularizers.L2(0.02)))
# model_.add(tf.keras.layers.Dropout(0.5))
# model_.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_constraint=tf.keras.constraints.MaxNorm(0.1), kernel_regularizer=tf.keras.regularizers.L2(0.02)))
model_.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                 kernel_regularizer=tf.keras.regularizers.L2(0.02)))
model_.add(tf.keras.layers.Dropout(0.4))
model_.add(tf.keras.layers.Dense(units=2, activation='softmax'))
# model_.build()
model_.summary()


"""End Training with base model params set to not trainable"""

start_time = datetime.datetime.now()

datetime_tag = start_time.strftime("%Y%m%d-%H%M%S")

base_model_.trainable = True

model_.summary()


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def scheduler2(epoch, lr):
    if epoch < 50:
        return lr
    else:
        # elif epoch%10 == 0:
        return lr * tf.math.exp(-0.1)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler2, verbose=1)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003, epsilon=5e-9, beta_1=0.9, beta_2=0.999)
model_.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               optimizer=adam_optimizer,
               metrics=['accuracy', ])
early_stop = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000001)

log_dir = "final_training/tfb_logs/fit/" + datetime_tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
hist_ = model_.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=100,
                   callbacks=[tensorboard_callback])
                   # callbacks=[tensorboard_callback, lr_scheduler])



print(type(hist_), hist_)
# hist_.history['accuracy']

plot_acc(hist_)
plot_loss(hist_)

print("Evaluation with test data", model_.evaluate(X_test, y_test))

finish_time = datetime.datetime.now()

print("Time delta =", (finish_time - start_time))

hist_df = pd.DataFrame(hist_.history)
hist_csv_file = f'final_training/history_augm_{datetime_tag}.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    print("CSV history saved")

model_.save(f"final_training/saved/final_model_augm_30s_clean_balanced_12_{datetime_tag}")

with open(f'final_training/hist_augm_{datetime_tag}.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(hist_.history, file)
    print("History file saved")

# model_.save("final_model")

test_predictions = model_.predict(X_test)

with open(f'final_training/preds_augm_{datetime_tag}.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(test_predictions, file)
    print("test_predictions file saved")

print("-" * 25, "DONE", "-" * 25)