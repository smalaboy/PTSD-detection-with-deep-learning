# -*- coding: utf-8 -*-
"""v2 final vggish_finetune_with_keras_tuner.ipynb

Automatically generated by Colaboratory.

Original file is located at
	https://colab.research.google.com/drive/1AxsnIQGNtv6ppyomy9rLSg5KI1yPSIpL
"""

# !pip list

# !zip -r data.zip /content/data/*

# !cat vggish_params.py

# !unzip ./data.zip

# !git clone https://github.com/DTaoo/VGGish.git
# !cp VGGish/* .
# !pip install numpy resampy tensorflow tf_slim six soundfile

# !pip install -q -U keras-tuner

# !unzip '/content/drive/MyDrive/PTSD dataset/saved_ds.zip'

from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
import librosa
import os
import librosa.display
import resampy
import datetime
from posixpath import dirname
from sklearn.model_selection import train_test_split
import audioread
# import keras
# from preprocess_sound import preprocess_sound
from keras import backend as K
import keras_tuner as kt
import matplotlib.pyplot as plt

import mel_features
import vggish_params
import vggish_params as params

import random
import sys

# Params
# path = '/content/drive/MyDrive/PTSD dataset' # Base data path # Should contain folders yes_ptsd and no_ptsd

FOLD_0_TRAIN_DATA_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/folds_files/fold_0_train"
FOLD_0_TEST_DATA_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/folds_files/fold_0_test"

FOLD_1_TRAIN_DATA_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/folds_files/fold_1_train"
FOLD_1_TEST_DATA_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/folds_files/fold_1_test"

FOLD_2_TRAIN_DATA_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/folds_files/fold_2_train"
FOLD_2_TEST_DATA_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/folds_files/fold_2_test"

FOLD_0_TRAIN_SAVED_DS_PATH = '/raid/home/labusermoctar/ptsd_dataset/final_data/folds_saved_ds/fold_0/train_ds_b30s'
FOLD_0_TEST_SAVED_DS_PATH = '/raid/home/labusermoctar/ptsd_dataset/final_data/folds_saved_ds/fold_0/test_ds_b30s'

FOLD_1_TRAIN_SAVED_DS_PATH = '/raid/home/labusermoctar/ptsd_dataset/final_data/folds_saved_ds/fold_1/train_ds_b30s'
FOLD_1_TEST_SAVED_DS_PATH = '/raid/home/labusermoctar/ptsd_dataset/final_data/folds_saved_ds/fold_1/test_ds_b30s'

FOLD_2_TRAIN_SAVED_DS_PATH = '/raid/home/labusermoctar/ptsd_dataset/final_data/folds_saved_ds/fold_2/train_ds_b30s'
FOLD_2_TEST_SAVED_DS_PATH = '/raid/home/labusermoctar/ptsd_dataset/final_data/folds_saved_ds/fold_2/test_ds_b30s'

# VGGish model weigths paths
WEIGHTS_PATH = '/raid/home/labusermoctar/ptsd_audio_detection/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = '/raid/home/labusermoctar/ptsd_audio_detection/vggish_audioset_weights.h5'

# Checkpoints
CHECKPOINT_FILEPATH = '/raid/home/labusermoctar/ptsd_audio_detection/final_models/checkpoints'
SAVED_MODELS_PATH = "/raid/home/labusermoctar/ptsd_audio_detection/final_models/saved"

# Epochs
EPOCHS = 100

# FLAG
IS_COLAB = False

SEED = 42

AUTO_BALANCE = True

classes = ["Without PTSD", "With PTSD"]

EXAMPLE_WINDOW_SECONDS = 29.76
EXAMPLE_HOP_SECONDS = 29.76
NUM_FRAMES = 2976

BATCH_SIZE = 5

USE_SAVED_DATASET = False

INPUT_SHAPE = (NUM_FRAMES, params.NUM_BANDS, 1)

tf.config.run_functions_eagerly(True)

"""Dataset loading and processing"""


def list_files_rec(path):
    files = os.listdir(path)
    files = [os.path.abspath(os.path.join(path, f)) for f in files]
    dirs = [f for f in files if os.path.isdir(f)]
    if not dirs:
        return files
    files = list(set(files) - set(dirs))
    for dir in dirs:
        files += list_files_rec(dir)
    return files


def load_file_and_build_labels(path, auto_balance=True, shuffle=True):
    ptsd = list_files_rec(os.path.join(path, 'yes'))
    no_ptsd = list_files_rec(os.path.join(path, 'no'))
    if auto_balance:
        LENGTH_YES = LENGTH_NO = min(len(ptsd), len(no_ptsd))
    else:
        LENGTH_YES = len(ptsd)
        LENGTH_NO = len(no_ptsd)
    ptsd = ptsd[:LENGTH_YES]
    no_ptsd = no_ptsd[:LENGTH_NO]
    filenames = ptsd + no_ptsd
    labels = [1 for _ in range(LENGTH_YES)] + [0 for _ in range(LENGTH_NO)]
    if shuffle:
        zipped_files_labels = list(zip(filenames, labels))
        random.shuffle(zipped_files_labels)
        filenames = [x for x, _ in zipped_files_labels]
        labels = [y for _, y in zipped_files_labels]
    print("For path : ", path, len(filenames), "files")
    return filenames, labels


def preprocess_sound(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.
  Args:
	data: np.array of either one dimension (mono) or two dimensions
	  (multi-channel, with the outer dimension representing channels).
	  Each sample is generally expected to lie in the range [-1.0, +1.0],
	  although this is not required.
	sample_rate: Sample rate of data.
  Returns:
	3-D np.array of shape [num_examples, num_frames, num_bands] which represents
	a sequence of examples, each of which contains a patch of log mel
	spectrogram, covering num_frames frames of audio and num_bands mel frequency
	bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        # vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        # vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
        EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    return log_mel_examples


# def make_dataset(files=[], labels=[]):
#   temp = []
#   for f in files:
#     print(f)
#     data, sr = librosa.load(f)
#     temp.append(preprocess_sound(data, sr))
#   return tf.ragged.constant(temp).to_tensor(), labels

def noise_injection(sig, sr, alpha=0.01):
    return sig - alpha*np.random.random(size=sig.shape)

def pitch_change(sig, sr, pitch_factor):
    return librosa.effects.pitch_shift(sig, sr, pitch_factor)

def shift_time(sig, sr, shift_max_sec, shift_direction="both"):
    shift = np.random.randint(sr * shift_max_sec)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(sig, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def augment_audio_sig(sig, sr, noise_alpha=0.01, pitch_factor=0.5, shift_by_sec=2, shift_direction="both"):
    y = sig
    y = noise_injection(sig, sr, alpha=noise_alpha)
    y = pitch_change(y, sr, pitch_factor=pitch_factor)
    y = shift_time(y, sr,  shift_by_sec, shift_direction)
    return y

def augment_audio_file(path):
    sig, sr = librosa.load(path)
    return augment_audio_sig(sig, sr)


def make_dataset(files=[], labels=[], reduce_mean=True, augment_data=False, noise_alpha=0.01, pitch_factor=0.5, shift_by_sec=2, shift_direction="both"):
    '''
    Build full dataset features
    @params:
        files: files list
        lables: labels list
        reduce_mean: reduce features on axis 0
        augment_data: use data augmentation
        noise_alpha: for noise injection
        pitch factor: for pitch change
        shift_by_sec: for time shifting
        shift_direction: direction of time shifting
    @returns: tuple of (list_of_features, list_of_labels)
    '''
    temp = []
    updated_labels = labels
    print("Data augmentation is", augment_data)
    for i, f in enumerate(files):
        try:
            # print(f)
            data, sr = librosa.load(f)
            if augment_data:
                data = augment_audio_sig(data, sr, noise_alpha=noise_alpha, pitch_factor=pitch_factor, shift_by_sec=shift_by_sec, shift_direction=shift_direction)
            p_data = preprocess_sound(data, sr)
            if reduce_mean:
                p_data = tf.reduce_mean(np.expand_dims(p_data, axis=3), axis=0)
            else:
                (a, b, c) = p_data.shape
                p_data = np.reshape(p_data, (a * b, c))
                p_data = np.expand_dims(p_data, axis=2)
            print(p_data.shape)
            # print("-"*40)
            temp.append(p_data)
        except RuntimeError as ex:
            print("File: ", f, " .RuntimeError:", ex)
            updated_labels.pop(i)
        except audioread.exceptions.NoBackendError as ex:
            print("File: ", f, " .NoBackendError:", ex)
            updated_labels.pop(i)
    out = tf.convert_to_tensor(temp), labels
    print(out[0].shape)
    return out


def signals_to_mel_spect(signals=[], labels=[], reduce_mean=False):
    '''
    Make mel spectograms from audio signals
    '''
    out = []
    for (sig, sr) in signals:
        _ = preprocess_sound(sig, sr)
        print("_____", _.shape)
        if not reduce_mean:
            (a, b, c) = _.shape
            _ = np.reshape(_, (a * b, c))
            _ = np.expand_dims(_, axis=2)
        else:
            _ = tf.reduce_mean(np.expand_dims(_, axis=3), axis=0)
        out.append(_)
    # out = [preprocess_sound(sig, sr) for (sig, sr) in signals]
    out = tf.convert_to_tensor(out)
    print("out", out.shape)
    return out, labels


clock_start = datetime.datetime.now()

n_alpha = 0.01
p_factor = 0.5
shift_secs = 2
shift_direc = "both"
augm_data = False

fold_0_X, fold_0_y = load_file_and_build_labels(FOLD_0_TRAIN_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
print("fold_0 data size", len(fold_0_X), len(fold_0_y))

fold_0_ds = tf.data.Dataset.from_tensor_slices(make_dataset(
    fold_0_X, fold_0_y, reduce_mean=False, augment_data=augm_data, noise_alpha=n_alpha,
    pitch_factor=p_factor, shift_by_sec=shift_secs, shift_direction=shift_direc
))
# full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, labels))

fold_0_ds = fold_0_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf.data.experimental.save(fold_0_ds, FOLD_0_TRAIN_SAVED_DS_PATH)
print("-------------------------------------- fold_0 Train ds 1 OK --------------------------------------------------------------------")


fold_0_test_X, fold_0_test_y = load_file_and_build_labels(FOLD_0_TEST_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
print("fold_test_0 data size", len(fold_0_test_X), len(fold_0_test_y))

fold_0_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(
    fold_0_test_X, fold_0_test_y, reduce_mean=False, augment_data=augm_data, noise_alpha=n_alpha,
    pitch_factor=p_factor, shift_by_sec=shift_secs, shift_direction=shift_direc
))
# full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, labels))

fold_0_test_ds = fold_0_test_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf.data.experimental.save(fold_0_test_ds, FOLD_0_TEST_SAVED_DS_PATH)
print("-------------------------------------- fold_test_0 Train ds 1 OK --------------------------------------------------------------------")


fold_1_X, fold_1_y = load_file_and_build_labels(FOLD_1_TRAIN_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
print("fold_1 data size", len(fold_1_X), len(fold_1_y))

fold_1_ds = tf.data.Dataset.from_tensor_slices(make_dataset(
    fold_1_X, fold_1_y, reduce_mean=False, augment_data=augm_data, noise_alpha=n_alpha,
    pitch_factor=p_factor, shift_by_sec=shift_secs, shift_direction=shift_direc
))
# full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, labels))

fold_1_ds = fold_1_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf.data.experimental.save(fold_1_ds, FOLD_1_TRAIN_SAVED_DS_PATH)
print("-------------------------------------- fold_1 Train ds 1 OK --------------------------------------------------------------------")


fold_1_test_X, fold_1_test_y = load_file_and_build_labels(FOLD_1_TEST_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
print("fold_test_0 data size", len(fold_1_test_X), len(fold_1_test_y))

fold_1_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(
    fold_1_test_X, fold_1_test_y, reduce_mean=False, augment_data=augm_data, noise_alpha=n_alpha,
    pitch_factor=p_factor, shift_by_sec=shift_secs, shift_direction=shift_direc
))
# full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, labels))

fold_1_test_ds = fold_1_test_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf.data.experimental.save(fold_1_test_ds, FOLD_1_TEST_SAVED_DS_PATH)
print("-------------------------------------- fold_1_testds 1 OK --------------------------------------------------------------------")


fold_2_X, fold_2_y = load_file_and_build_labels(FOLD_2_TRAIN_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
print("fold_2 data size", len(fold_2_X), len(fold_2_y))

fold_2_ds = tf.data.Dataset.from_tensor_slices(make_dataset(
    fold_2_X, fold_2_y, reduce_mean=False, augment_data=augm_data, noise_alpha=n_alpha,
    pitch_factor=p_factor, shift_by_sec=shift_secs, shift_direction=shift_direc
))
# full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, labels))

fold_2_ds = fold_2_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf.data.experimental.save(fold_2_ds, FOLD_2_TRAIN_SAVED_DS_PATH)
print("-------------------------------------- fold_2 Train ds 1 OK --------------------------------------------------------------------")


fold_2_test_X, fold_2_test_y = load_file_and_build_labels(FOLD_2_TEST_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
print("fold_test_0 data size", len(fold_2_test_X), len(fold_2_test_y))

fold_2_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(
    fold_2_test_X, fold_2_test_y, reduce_mean=False, augment_data=augm_data, noise_alpha=n_alpha,
    pitch_factor=p_factor, shift_by_sec=shift_secs, shift_direction=shift_direc
))
# full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, labels))

fold_2_test_ds = fold_2_test_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf.data.experimental.save(fold_2_test_ds, FOLD_2_TEST_SAVED_DS_PATH)
print("-------------------------------------- fold_2_testds 1 OK --------------------------------------------------------------------")













#test_X, test_y = load_file_and_build_labels(TEST_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
#print("Test data size", len(test_X), len(test_y))

#test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_X, test_y, reduce_mean=False))

#test_ds = test_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#tf.data.experimental.save(test_ds, TEST_DS_PATH)
print("-------------------------------------- Test ds OK --------------------------------------------------------------------")

#val_X, val_y = load_file_and_build_labels(VAL_DATA_PATH, auto_balance=AUTO_BALANCE, shuffle=True)
#print("Val data size", len(val_X), len(val_y))

#val_ds = tf.data.Dataset.from_tensor_slices(make_dataset(val_X, val_y, reduce_mean=False))

#val_ds = val_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#tf.data.experimental.save(val_ds, VAL_DS_PATH)
print("-------------------------------------- Val ds OK --------------------------------------------------------------------")

# if not USE_SAVED_DATASET:
#   full_test_ds = tf.data.Dataset.from_tensor_slices(make_dataset(test_filesnames, test_labels))

# test_ds = full_test_ds.cache().enumerate().filter(lambda x, y: x % 2 == 0).map(lambda x, y: y)
# val_ds = full_test_ds.cache().enumerate().filter(lambda x, y: x % 2 != 0).map(lambda x, y: y)

#train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#val_ds = val_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#test_ds = test_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# print(len(list(full_test_ds.as_numpy_iterator())))

#tf.data.experimental.save(train_ds, TRAIN_DS_PATH)
#tf.data.experimental.save(test_ds, TEST_DS_PATH)
#tf.data.experimental.save(val_ds, VAL_DS_PATH)

clock_end = datetime.datetime.now()

print("+" * 25, "Data processing finished", "+" * 25)
print(clock_start, clock_end, "duration=", (clock_end - clock_start))
