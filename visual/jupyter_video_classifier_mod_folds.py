# -*- coding: utf-8 -*-
"""jupyter_video_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1usTtkLZnA11N_xpMiBr2GLuPwscLRHih
"""



# !unzip /content/drive/MyDrive/datasets/extended_ptsd_npy_ds.zip
# !pip install keras-tuner -q
# !nvidia-smi

#import pandas as pd

# path = '/content/train.csv'

# dfObj = pd.read_csv(path)

# indexNames = dfObj[dfObj['class']==0].index
# dfObj.drop(indexNames , inplace=True)

# dfObj.to_csv(path)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTM, TimeDistributed, Softmax, BatchNormalization, Dropout, GRU
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
# import numpy as np
# from sklearn.model_selection import GridSearchCV, KFold
import cv2
import pandas as pd
import numpy as np
# from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
# from google.colab.patches import cv2_imshow
# import numpy as np
import os
#import keras_tuner
import datetime
# from sklearn import metrics
# from sklearn import model_selection
# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(32)


from sklearn.metrics import classification_report


# SEED = 42


FOLDS_TRAIN = ["train_fold_1.csv", "train_fold_2.csv", "train_fold_3.csv", ]
FOLDS_TEST = ["test_fold_1.csv", "test_fold_2.csv", "test_fold_3.csv", ]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from kerastuner.applications import HyperResNet
# from kerastuner.tuners import Hyperband
# hypermodel = HyperResNet(input_shape=(28, 28, 1), classes=10)

# tuner = Hyperband(
#     hypermodel,
#     objective='val_accuracy',
#     max_trials=20,
#     directory='FashionMnistResNet',
#     project_name='FashionMNIST')

# tuner.search(train_images, train_labels_binary, validation_split=0.1)

def create_model(lstm_units, epsilon, lr, dropout, optimizer):
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
      if isinstance(layer, BatchNormalization):
          layer.trainable = True
      else:
          layer.trainable = False

  model = tf.keras.Sequential()
  model.add(TimeDistributed(base_model, input_shape=(None, 299, 299, 3)))
  # model.add(Dropout(dropout))
  # model.add(Dense(100))
  # kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32)
  model.add(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout))
  # model.add(Dropout(dropout))
  # model.add(Dense(hp.Int("dense_units", min_value=100, max_value=500, step=200)))
  model.add(Dense(2))
  # model.add(LSTM(64))

  model.add(Softmax())
  model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
  return model

def build_model(hp):
    # lstm_units = hp.Choice("lstm_units", [2, 4, 8, 16, 32, 64])
    #epsilon = hp.Choice("epsilon", [0., 0.2, 0.1, 0.9])
    #lr = hp.Choice("lr", [0.01, 0.001, 3e-4, 0.1])
    # dropout = hp.Choice("dropout", [0.2, 0.3, 0.4, 0.5, 0.25, 0.35, 0.45])
    lstm_units = 64
    epsilon = 0
    lr = 3e-4
    dropout = 0.5
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=epsilon)
    #optimizer = hp.Choice("optimizer", ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"])
    
    model = create_model(
        lstm_units=lstm_units, epsilon=epsilon, lr=lr, dropout=dropout, optimizer=optimizer
    )
    return model

#tuner = keras_tuner.RandomSearch(
#    hypermodel=build_model,
#    objective="val_accuracy",
#    max_trials=10,
#    directory='/tmp/tb_logs',
#    project_name='two'
#)
# tuner = keras_tuner.tuners.SklearnTuner(
#     oracle=keras_tuner.oracles.BayesianOptimizationOracle(
#         objective=keras_tuner.Objective('val_f1', 'max'),
#         max_trials=20),
#     hypermodel=build_model,
#     scoring=metrics.make_scorer(metrics.accuracy_score),
#     cv=model_selection.StratifiedKFold(5),
#     directory='.',
#     project_name='my_project')
#tuner.search_space_summary()



# build_model(keras_tuner.HyperParameters())

# tuner.search()

# print(model(train_s.__getitem__(0)[0]).shape)

# detector = MTCNN()
# df = pd.read_csv("/content/PTSD-FromVideo/dataset.csv")
# print(df["class"])

def crop_face(img):
    # print('face cropping started')
    result = detector.detect_faces(img)
    feature_arr = None
    for i in result:
        crop_img = img[i["box"][1]:i["box"][1]+i["box"][3], i["box"][0]:i["box"][0]+i["box"][2]]
        crop_img = cv2.resize(crop_img, (299, 299))
        # X = preprocess_input(crop_img)
        # X = np.expand_dims(X, axis=0)
        # features = base_model.predict(X)
        # print(features.shape)
        # print('Predicted:', decode_predictions(features, top=3)[0])
        # cv2.imshow(crop_img)
        # print("crop_img size")
        crop_img = tf.expand_dims(crop_img, 0)
        # print(crop_img.shape)
        if feature_arr is None:
            feature_arr = crop_img
        else:
            feature_arr = tf.concat([feature_arr, crop_img], axis=0)
        # feature_arr.append(features)
            # print(feature_arr.shape)
    # if feature_arr is not None:
    #     feature_arr = tf.expand_dims(feature_arr, 0)
    # print('face cropped')
    return feature_arr

def load_video(path, max_frames=60):
    # print('video loading started')
    frames = None
    # print(path)
    cap = cv2.VideoCapture(path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_step = total_frames//max_frames
    count = 0
    # try:
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("F")
    #             break
    #         frame = tf.expand_dims(frame, 0)
    #         if count%frames_step == 0:

    #               if frames is None:
    #                   frames = frame
    #               else:
    #                   frames = tf.concat([frames, frame], axis=0)
    #         count+=1
            
    #         # pass it through resnet
    #         # make a vector
    #         if frames != None:
    #             if len(frames) >= max_frames:
    #                 # cv2.imshow("cap", frame)
    #                 break
    # finally:
    #     cap.release()
    # return tf.expand_dims(frames, 0)
    for i in range(max_frames):
      cap.set(1,i*frames_step)
      success,img = cap.read() 
      # print(success, img.shape)
      img = cv2.resize(img, (299, 299))
      # if not count:
      #   cv2_imshow(img)
      img = tf.expand_dims(img, 0)
      
      count+=1
      if frames is None:
        frames = img
      else:
        frames = tf.concat([frames, img], axis=0)

    cap.release()
    
    return tf.expand_dims(frames, 0)

# !pip install --upgrade  cupy-cuda111
# import cupy as cp
# import numpy as np

# !nvcc --version

# cp1 = cp.load('/content/v1.npy')
# # print(cp1[0][0].shape)
# cap = cp1.toDlpack()
# b = tf.experimental.dlpack.from_dlpack(cap)
# # model(b)
# # print(model(b))

# cp1[0][0]

# Commented out IPython magic to ensure Python compatibility.
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, path, batch_size, folder):
        self.df = pd.read_csv(path)
        # print(df["class"])
        self.n = len(self.df)
        self.batch_size = batch_size
        self.folder = folder
        # self.shuffle = True
    # PTSD-FromVideo
    def on_epoch_end(self):
        # if self.shuffle:
        self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, index):
        # return df["name"][index], df["class"][index]
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        # X, y = self
        X = None
        Y = []
        for i in range(len(batches)):
            # print(batches)
            file_name = batches["name"][index*self.batch_size+i]
            file_name = file_name[:-3]

            # print(file_name)
            file_name = "{}".format(self.folder)+file_name+"npy"
            file_data = np.load(file_name, allow_pickle=True)
            # cap = file_data.toDlpack()
            # print(file_name)
            # print(file_data)
            file_data_tf = tf.convert_to_tensor(file_data)

            if X == None:
                X = file_data_tf
            else:
                X = tf.concat([X, file_data_tf], axis=0)
            # print(X.shape)
            Y.append(batches["class"][index*self.batch_size+i])
            
        return X/255, tf.convert_to_tensor(Y)

    
    def __len__(self):
        return self.n // self.batch_size


def schedule(epoch, lr):
    if epoch>=70 and epoch%10 == 0:
        return lr * 0.3
    return lr


scores = []

# Training
for i in [1, 2, 3]:

    # train_ds = CustomDataGen("dataset.csv", 2)
    train_s = CustomDataGen(FOLDS_TRAIN[i-1], 6, "new_npy")
    test_s = CustomDataGen(FOLDS_TEST[i-1], 4, "new_npy")
    # print(X.shape)

    # %load_ext tensorboard

    # %tensorboard --logdir /tmp/tb_logs

    # model.build(input_shape = (299, 299, 3))
    # model = KerasClassifier(build_fn=create_model, verbose=0)
    # Define the grid search parameters
    # batch_size = [10,20,40]
    # epochs = [10,50,100]
    # Make a dictionary of the grid search parameters
    # param_grid = dict(batch_size = batch_size,epochs = epochs)
    # grid = GridSearchCV(estimator = model,param_grid = param_grid,cv = KFold(),verbose = 10)
    # grid_result = grid.fit(train_s,train_s)
    # print(model.get_params().keys())

    checkpoint_filepath = f'tmp/checkpoint_fold_{i}'
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='accuracy',
        mode='max',
        save_best_only=True)
    #tuner.search(train_s, epochs=10, validation_data=val_s, callbacks=[model_checkpoint_callback, stop_early, tf.keras.callbacks.TensorBoard("/tmp/tb_logs")])
    # callbacks=[]
    # tuner.search(train_s)
    # history = model.fit(train_s, validation_data=val_s, epochs=20, callbacks=[model_checkpoint_callback])
    # # print(history)

    #tuner.results_summary()

    #best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    # print(best_hps)
    # model = create_model(13, 1.0, 0.0003)
    # print(best_hps.get('lstm_units'), best_hps.get('lr'), best_hps.get('epsilon'), best_hps.get('dropout'))
    """print(best_hps.get('epsilon'))
    model = tuner.hypermodel.build(best_hps)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)
    history1 = model.fit(train_s, epochs=20, validation_data=val_s, callbacks=[model_checkpoint_callback, stop_early])
    
    val_acc_per_epoch = history1.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    hypermodel = tuner.hypermodel.build(best_hps)
    
    # Retrain the model
    history = hypermodel.fit(train_s, epochs=best_epoch, validation_data=val_s, callbacks=[model_checkpoint_callback])"""

    # hypermodel = tuner.hypermodel.build(best_hps)
    # # Retrain the model
    # history = hypermodel.fit(train_s, epochs=best_epoch, validation_data=val_s, callbacks=[model_checkpoint_callback])

    """lstm_units = 64
        epsilon = 0
        lr = 3e-4
        dropout = 0.5
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=epsilon)"""

    model = create_model(lstm_units=64, epsilon=0.1, lr=3e-4, dropout=0.5, optimizer=tf.keras.optimizers.SGD(learning_rate=3e-4, momentum=0.1))
    checkpoint_filepath = f'tmp/checkpoint_fold_{i}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    log_dir = f"tfb_logs/fit_fold_{1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history1 = model.fit(train_s, epochs=200, callbacks=[model_checkpoint_callback, tensorboard_callback, lr_scheduler])
    test_history = model.evaluate(test_s)
    test_df = pd.DataFrame(test_history)
    hist_csv_file = f'test_hist/fold_{i}.csv'
    os.makedirs('test_hist', exist_ok=True)
    with open(hist_csv_file, mode='w') as f:
        test_df.to_csv(f)

    hist_df = pd.DataFrame(history1.history)
    hist_csv_file = f'history/fold_{i}.csv'
    os.makedirs('history', exist_ok=True)
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    print(history1.history.keys())
    import matplotlib.pyplot as plt

    os.makedirs('fold_plots', exist_ok=True)
    # summarize history for accuracy
    plt.figure()
    plt.plot(history1.history['accuracy'])
    # plt.plot(history1.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(f'fold_plots/acc_fold_{i}.png')

    # summarize history for loss
    plt.figure()
    plt.plot(history1.history['loss'])
    # plt.plot(history1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(f'fold_plots/loss_fold_{i}.png')

    os.makedirs('saved_models', exist_ok=True)
    model.save(f'saved_models/ptsd_model_video_{i}.h5')

    print(f"---------------------------------------------------------- Model fold {i} saved -------------------------------------------------------------------")

