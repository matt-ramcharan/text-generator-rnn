from __future__ import print_function
import matplotlib.pyplot as plt
#import numpy as np
import keras.callbacks
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model
import argparse
from RNN_utils import *
#from keras_tqdm import TQDMCallback
#from tqdm import tqdm
import pickle
from tensorflow.python.client import device_lib

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.66
# set_session(tf.Session(config=config))

class My_Callback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("\n")
    gen = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    print(gen, end="\n")
    print('\n\nEpoch: {}\n'.format(epoch), file=open("Pott.txt", "a"), end="\n")
    print(gen, file=open("Pott.txt", "a"), end="\n")
    model.save('./HPweights/checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, epoch))

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/test.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
epoch = args['nb_epoch']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

## Generate some sample before training to know how bad it is!
#generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char)

# if not WEIGHTS == '':
#   model.load_weights(WEIGHTS)
#   #epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
#   generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)

# Training if there is no trained weights specified
call = My_Callback()
print(device_lib.list_local_devices())



if args['mode'] == 'train' or WEIGHTS == '':
    # Creating training data
    X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

    # Creating and compiling the Network
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True,dropout=0.25))
    for i in range(LAYER_NUM - 1):
      model.add(LSTM(HIDDEN_DIM, return_sequences=True, dropout=0.25))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    #try:
        #pickle.dump(X, open("./HPweights/X.p", "wb"),protocol=4)
        #pickle.dump(y, open("./HPweights/y.p", "wb"),protocol=4)
        #pickle.dump(VOCAB_SIZE, open("./HPweights/VOCAB_SIZE.p", "wb"),protocol=4)
        #pickle.dump(ix_to_char, open("./HPweights/ix_to_char.p", "wb"),protocol=4)
    #except FileNotFoundError:
    #    print("Cannot Pickle")


    model.summary()
    print('\n\nEpoch: {}\n'.format(epoch))

    hist = model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=epoch, validation_split=0.05, callbacks=[call])

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss" +
                "Batch"+str(BATCH_SIZE)+
                "HiddenDim"+str(HIDDEN_DIM)+
                "LayerNum" + str(LAYER_NUM) +
                "SeqLength"+str(SEQ_LENGTH)+
                "Epochs"+str(epoch)+
                ".pdf", format="pdf")
    plt.show()
    #epoch += 1
    plot_model(model, to_file='model.png', show_shapes=True)

    print(generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char))

    #if epoch % 10 == 0:
    #model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS != '':
  # Loading the trained weights
  #X = pickle.load(open("./HPweights/X.p", "rb"))
  #y= pickle.load(open("./HPweights/y.p", "rb"))

  VOCAB_SIZE = pickle.load(open("./40EpochHPRun/HPweights/VOCAB_SIZE.p", "rb"))
  ix_to_char = pickle.load(open("./40EpochHPRun/HPweights/ix_to_char.p", "rb"))

  model = load_model(WEIGHTS)
  text = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  print(text,end ="\n")
  print(text, file=open("GenPott.txt", "a"), end="\n")
  print('\n\n')
else:
  print('\n\nNothing to do!')

plot_model(model, to_file='model.png')




