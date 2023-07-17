import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
import librosa
import os
import librosa
import librosa.display
import resampy
import datetime
from posixpath import dirname
from sklearn.model_selection import train_test_split
import audioread
# import keras
# from preprocess_sound import preprocess_sound
# import keras_tuner as kt
import matplotlib.pyplot as plt

import mel_features
import vggish_params as params


# In[3]:


import random
import sys
import mel_features
import vggish_params
import cv2
import pandas as pd
# from tensorflow.keras.preprocessing import image
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, TFBertModel, BertConfig


# In[60]:


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

BATCH_SIZE = 32

USE_SAVED_DATASET = False

INPUT_SHAPE = (NUM_FRAMES, params.NUM_BANDS, 1)

AUDIO_MODEL_INPUT_SHAPE = (NUM_FRAMES, params.NUM_BANDS, 1)

tf.random.set_seed(SEED)

configuration = BertConfig()  # default parameters and configuration for BERT

tf.config.run_functions_eagerly(True)


# In[5]:


import tensorflow as tf

print("tf cuda", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print("tf physical devices", tf.config.list_physical_devices("GPU"))


# In[6]:

# tf_config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=tf_config)

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


# In[68]:


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


# In[69]:


def make_audio_features(path):
    files = list_files_rec(path)
    sigs = []
    for f in files:
        data, sr = librosa.load(f)
        sigs.append(data)
    np_sig = np.concatenate(sigs)
    # print(np_sig.shape)
    p_data = preprocess_sound(np_sig, sr)
    # print(p_data.shape)
    p_data = tf.reduce_mean(np.expand_dims(p_data, axis=3), axis=0)
    print(p_data.shape, path)
    return p_data


# In[70]:


def make_audio_dataset(path_list = []):
    temp = []
    for path in path_list:
        features = make_audio_features(path)
        temp.append(features)
    return tf.convert_to_tensor(temp)


# In[92]:


# # make_audio_features("/raid/home/labusermoctar/ptsd_dataset/final_data/splits/train/yes/War veterans/vidm108")
# audio = make_audio_dataset(["/raid/home/labusermoctar/ptsd_dataset/final_data/splits/train/yes/War veterans/vidm108", "/raid/home/labusermoctar/ptsd_dataset/final_data/splits/train/yes/War veterans/vidm110", "/raid/home/labusermoctar/ptsd_dataset/final_data/splits/train/yes/War veterans/vidm112"])
# audio.shape


# In[71]:


def make_visual_features(file_path):
    path = file_path[:-3] + "npy"
    file_data = np.load(path, allow_pickle=True)
    file_data_tf = tf.convert_to_tensor(file_data)
    return tf.squeeze(file_data_tf, axis=0)/255


# In[72]:


def make_visual_dataset(path_list = []):
    temp = []
    for path in path_list:
        features = make_visual_features(path)
        temp.append(features)
    return tf.convert_to_tensor(temp)


# In[93]:


# # X = make_visual_features("/raid/home/labusermoctar/ptsd_images_detection/new_npy/train/PTSD/War veterans/vidm008.npy")
# visual = make_visual_dataset(["/raid/home/labusermoctar/ptsd_images_detection/new_npy/train/PTSD/War veterans/vidm008.npy", "/raid/home/labusermoctar/ptsd_images_detection/new_npy/train/PTSD/War veterans/vidm110.npy","/raid/home/labusermoctar/ptsd_images_detection/new_npy/train/PTSD/War veterans/vidm112.npy", ])
# print(visual.shape)


# In[73]:


class TokenizerLayer(tf.keras.layers.Layer):
  def __init__(self, tokenizer) -> None:
      super(TokenizerLayer, self).__init__()
      self.tokenizer = tokenizer

  def call(self, inputs):
    # print("inputs", list(inputs))
    out = []
    out_dict = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
    }
    for text in inputs:
      # print('text', text)
      value = str(tf.keras.backend.get_value(text)).lower()
      # print("value", value)
      tokens = self.tokenizer.encode_plus(
          value, 
          return_tensors='tf',
          truncation=True, 
          padding="max_length",
          max_length=512,
          )
      tokens["input_ids"] = tf.squeeze(tokens["input_ids"])
      tokens["attention_mask"] = tf.squeeze(tokens["attention_mask"])
      tokens["token_type_ids"] = tf.squeeze(tokens["token_type_ids"])
      out.append(tokens)
      out_dict["input_ids"].append(tokens["input_ids"])
      out_dict["attention_mask"].append(tokens["attention_mask"])
      out_dict["token_type_ids"].append(tokens["token_type_ids"])
      # print("dict", type(tokens), tokens)
    out_dict["input_ids"] = tf.convert_to_tensor(out_dict["input_ids"])
    out_dict["attention_mask"] = tf.convert_to_tensor(out_dict["attention_mask"])
    out_dict["token_type_ids"] = tf.convert_to_tensor(out_dict["token_type_ids"])
    # print(out_dict["input_ids"])
    # return tf.convert_to_tensor(out)
    return out_dict


# In[74]:


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def make_text_features(text_list=[]):
    tokens = TokenizerLayer(tokenizer)(text_list)
    return tokens


# In[75]:


def make_text_dataset(path_list = []):
    temp = []
    for path in path_list:
        text = ''
        with open(path, 'r') as file:
            text = file.read()
        temp.append(text)
    return make_text_features(temp)



def get_source_inputs(*args, **kwargs):
  ...


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
        aud_input =  tf.keras.layers.Input(shape=input_shape, name='input_1')
    else:
        if not K.is_keras_tensor(input_tensor):
            aud_input =  tf.keras.layers.Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor



    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)



    if include_top:
        # FC block
        x = tf.keras.layers.Flatten(name='flatten_')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
        x = tf.keras.layers.Dense(out_dim, activation="relu", name='vggish_fc2')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)


    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    # Create model.
    model = tf.keras.models.Model(inputs, x, name='VGGish')


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


# In[24]:


def create_audio_model(hp=None, drop_last_layers=False):
  base_m = VGGish(load_weights=False, include_top=False, input_shape=INPUT_SHAPE, pooling='avg')
  base_m.load_weights(WEIGHTS_PATH)
  base_m.trainable = True

  if drop_last_layers:
    return base_m

  max_norm = 3
  kernel_reg = 0.02
  dropout_rate = 0.4
  learning_rate = 0.00003
  epsilon = 5e-9

  if hp:
    max_norm = hp.Choice('max_norm', values=[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.5,  0.7, 0.9])
    kernel_reg = hp.Choice('kernel_reg', values=[0.01, 0.02, 0.03, 0.04,0.06, 0.07, 0.09, 0.1, 0.2])
    dropout_rate = hp.Choice('dropout_rate', values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    learning_rate = hp.Choice('learning_rate', values=[0.00001, 0.00002, 0.00003, 0.00004, 0.00005])
    epsilon = hp.Choice('epsilon', values=[1e-7, 1e-8, 5e-9, 1e-9])

  m = tf.keras.models.Sequential()
  m.add(base_m)
  m.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_constraint=tf.keras.constraints.MaxNorm(max_norm), kernel_regularizer=tf.keras.regularizers.L2(kernel_reg)))
  # model_.add(tf.keras.layers.Dropout(0.5))
  m.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_constraint=tf.keras.constraints.MaxNorm(max_norm), kernel_regularizer=tf.keras.regularizers.L2(kernel_reg)))
  m.add(tf.keras.layers.Dropout(dropout_rate))
  m.add(tf.keras.layers.Dense(units=2, activation='softmax'))

  m.compile(
    loss = 'categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
    metrics=['accuracy',]
  )

  return m


# *Visual model*

# In[25]:


def create_visual_model():
  lstm_units = 64
  epsilon = 0
  lr = 3e-4
  dropout = 0.5

  base_model = tf.keras.applications.ResNet50V2(
      include_top=False,
      weights="imagenet",
      input_tensor=None,
      input_shape=(299, 299, 3),
      pooling='avg'
  )
  base_model.trainable=False
  # base_model.add(Dropout(dropout))
  for layer in base_model.layers:
      if isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.trainable = True
      else:
          layer.trainable = False

  base_model.summary()

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.TimeDistributed(base_model, input_shape=(None, 299, 299, 3)))
  model.add(tf.keras.layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout))

  return model


# *Textual model*

# In[26]:


class CustomBertModel(tf.keras.models.Model):
  def __init__(self, bert_model="bert-base-uncased", include_top=True):
    super(CustomBertModel, self).__init__()
    self.include_top = include_top
    self.bert_model = TFBertModel.from_pretrained(bert_model)
    self.dropout = tf.keras.layers.Dropout(0.1)
    self.fc = tf.keras.layers.Dense(2, activation=None)

  def call(self, inputs):
    # print(inputs)
    embeddings = self.bert_model(inputs).last_hidden_state
    embeddings = tf.math.reduce_mean(embeddings, axis=1)
    if not self.include_top:
      return embeddings
    embeddings = self.dropout(embeddings)
    logits = self.fc(embeddings)
    return logits


# In[27]:


def create_text_model():
   model = CustomBertModel(include_top=False)
   return model


# *Multimodal model*

# In[28]:


def create_multi_modal_model():
  
  text_input_ids = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
  text_input_attention_mask = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
  text_input_token_type_ids = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

  visual_input = tf.keras.layers.Input(shape=(60, 299, 299, 3))
  audio_input = tf.keras.layers.Input(shape=AUDIO_MODEL_INPUT_SHAPE)

  text_model = create_text_model()([text_input_ids, text_input_attention_mask, text_input_token_type_ids])
  text_model = tf.keras.layers.Dropout(0.2)(text_model)
  # text_model = tf.keras.models.Model(inputs=text_input, outputs=text_model)

  visual_model = create_visual_model()(visual_input)
  visual_model = tf.keras.layers.Dropout(0.2)(visual_model)
  # visual_model = tf.keras.models.Model(inputs=visual_input, outputs=visual_model)

  audio_model = create_audio_model(drop_last_layers=True)(audio_input)
  audio_model = tf.keras.layers.Dropout(0.2)(audio_model)
  # audio_model = tf.keras.models.Model(inputs=audio_input, outputs=audio_model)

  combined = tf.keras.layers.Concatenate()([text_model, visual_model, audio_model])

  outputs = tf.keras.layers.Dense(2, activation="softmax")(combined)

  return tf.keras.models.Model([text_input_ids, text_input_attention_mask, text_input_token_type_ids, visual_input, audio_input], outputs)


# In[29]:


model = create_multi_modal_model()


# In[30]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[31]:


# model([textual["input_ids"], textual["attention_mask"], textual["token_type_ids"], visual, audio])


# **Reading pandas df and building datasets**

# In[67]:


SAVED_TRAIN_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/train"
SAVED_TEST_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/test"
SAVED_VAL_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/val"

FOLD_0_TRAIN_SAVED_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/fold_0_train"
FOLD_0_TEST_SAVED_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/fold_0_test"

FOLD_1_TRAIN_SAVED_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/fold_1_train"
FOLD_1_TEST_SAVED_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/fold_1_test"

FOLD_2_TRAIN_SAVED_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/fold_2_train"
FOLD_2_TEST_SAVED_DS_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/saved_ds/multimodal/fold_2_test"


# In[32]:


df = pd.read_json("/raid/home/labusermoctar/ptsd_dataset/final_data/pandas_df/final_df.json")


# In[101]:


# df.columns


# In[34]:


df.head(5)


# In[35]:


TEXT_DATASET_BASE_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data/text_data"
VISUAL_DATASET_BASE_PATH = "/raid/home/labusermoctar/ptsd_images_detection/new_npy"
AUDIO_DATASET_BASE_PATH = "/raid/home/labusermoctar/ptsd_dataset/final_data"


# In[36]:


def _make_audio_full_path(row):
    part1 = f"fold_{0}"
    if row['test_fold'] == 0:
        part1 = f"{part1}_test"
    else:
         part1 = f"{part1}_train"
    return os.path.join(AUDIO_DATASET_BASE_PATH, f"folds_files/{part1}/{row['audio_path']}")

def _make_visual_full_path(row):
    return os.path.join(VISUAL_DATASET_BASE_PATH, f"{(row['vis_path'][1:])[:-3]}npy")

def _make_text_full_path(row):
    return os.path.join(TEXT_DATASET_BASE_PATH, row['text_path'])

df["full_audio_path"] = df.apply(lambda row: _make_audio_full_path(row), axis=1)
df["full_vis_path"] = df.apply(lambda row: _make_visual_full_path(row), axis=1)
df["full_text_path"] = df.apply(lambda row: _make_text_full_path(row), axis=1)


# In[103]:


# df["full_vis_path"][0]


# In[104]:


def make_full_dataset_from_df(df, shuffle=False, return_tf_dataset=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    i = 0
    audio_paths = df["full_audio_path"].tolist()
    visual_paths = df["full_vis_path"].tolist()
    text_paths = df["full_text_path"].tolist()
    labels = df["label"].tolist()
    
    visual_features = make_visual_dataset(visual_paths)
    text_features = make_text_dataset(text_paths)
    audio_features = make_audio_dataset(audio_paths)
    
    print("visual_features" ,visual_features.shape)
    print("audio_features", audio_features.shape)
    for k,v in text_features.items():
        print(k, v.shape)
    
    
    # for index, row in df.iterrows():
    #     # print(row)
    #     audio_path = row["full_audio_path"]
    #     vis_path = row["full_vis_path"]
    #     text_path = row["full_text_path"]
    #     label = row["label"]
    
    if return_tf_dataset:
        return tf.data.Dataset.from_tensor_slices(((text_features["input_ids"], text_features["attention_mask"], text_features["token_type_ids"], visual_features, audio_features), labels))
    return text_features["input_ids"], text_features["attention_mask"], text_features["token_type_ids"], visual_features, audio_features, labels
            
# train_ds = make_full_dataset_from_df(df.head(5), shuffle=True, return_tf_dataset=True)


# *TRAIN TEST VAL DS*

# In[52]:


def make_and_save_ds(df, save_path):
    if not save_path:
        return
    ds = make_full_dataset_from_df(df, shuffle=True, return_tf_dataset=True)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    tf.data.experimental.save(ds, save_path)
    return ds


# In[76]:


# Train_ds
start_time = datetime.datetime.now()
print(df[df.split=="train"].count())
train_ds = make_and_save_ds(df[df.split=="train"], SAVED_TRAIN_DS_PATH)
print("-"*10, "TRAIN DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[77]:


# test_ds
# start_time = datetime.datetime.now()
# print(df[df.split=="test"].count())
# test_ds = make_and_save_ds(df[df.split=="test"], SAVED_TEST_DS_PATH)
# print("-"*10, "TEST DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[78]:


# val_ds
# start_time = datetime.datetime.now()
# print(df[df.split=="val"].count())
# val_ds = make_and_save_ds(df[df.split=="val"], SAVED_VAL_DS_PATH)
# print("-"*10, "VAL DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[79]:


# fold_0_train_ds
start_time = datetime.datetime.now()
print(df[df.test_fold!=0].count())
fold_0_train_ds = make_and_save_ds(df[df.test_fold!=0], FOLD_0_TRAIN_SAVED_DS_PATH)
print("-"*10, "FOLD 0 TRAIN DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[80]:


# fold_1_train_ds
start_time = datetime.datetime.now()
print(df[df.test_fold!=1].count())
fold_1_train_ds = make_and_save_ds(df[df.test_fold!=1], FOLD_1_TRAIN_SAVED_DS_PATH)
print("-"*10, "FOLD 1 TRAIN DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[81]:


# fold_2_train_ds
start_time = datetime.datetime.now()
print(df[df.test_fold!=2].count())
fold_2_train_ds = make_and_save_ds(df[df.test_fold!=2], FOLD_2_TRAIN_SAVED_DS_PATH)
print("-"*10, "FOLD 2 TRAIN DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[82]:


# fold_0_test_ds
start_time = datetime.datetime.now()
print(df[df.test_fold==0].count())
fold_0_test_ds = make_and_save_ds(df[df.test_fold==0], FOLD_0_TEST_SAVED_DS_PATH)
print("-"*10, "FOLD 0 TEST DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[83]:


# fold_1_test_ds
start_time = datetime.datetime.now()
print(df[df.test_fold==1].count())
fold_1_test_ds = make_and_save_ds(df[df.test_fold==1], FOLD_1_TEST_SAVED_DS_PATH)
print("-"*10, "FOLD 1 TEST DS SAVED", (datetime.datetime.now() - start_time), "-"*10)


# In[84]:


# fold_2_test_ds
start_time = datetime.datetime.now()
print(df[df.test_fold==2].count())
fold_2_test_ds = make_and_save_ds(df[df.test_fold==2], FOLD_2_TEST_SAVED_DS_PATH)
print("-"*10, "FOLD 2 TEST DS SAVED", (datetime.datetime.now() - start_time), "-"*10)






