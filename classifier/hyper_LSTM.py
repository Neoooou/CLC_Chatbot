from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils,to_categorical
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras import optimizers
from classifier.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import numpy as np
from keras import callbacks
embedding_vector_length = 500
max_sequence_length = 18
vocab_size = None
def data():
    docs, y, links = preprocessing("../crawler/clcjsondata.txt").process()
    # integer encode documents
    t = Tokenizer()
    t.fit_on_texts(docs)
    X = t.texts_to_sequences(docs)
    global vocab_size
    vocab_size = len(t.word_index) + 1
    one_hot_y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_y, test_size=0.2, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=777)
    # truncate and pad input sequences
    max_sequence_length = 18
    X_train = sequence.pad_sequences(X_train, maxlen=max_sequence_length, padding="post")
    X_test = sequence.pad_sequences(X_test, maxlen=max_sequence_length, padding="post")
    X_val = sequence.pad_sequences(X_val, maxlen=max_sequence_length, padding="post")
    return X_train, X_val, X_test, y_train, y_val, y_test
def create_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    embedding_vector_length = 500
    model.add(layers.Embedding(vocab_size, embedding_vector_length, input_length=max_sequence_length))
    model.add(layers.LSTM(units={{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7)])}}, dropout={{uniform(0.5, 1)}},
                          recurrent_dropout={{uniform(0.5,1)}}))
    model.add(Dense(units=10, activation="softmax", kernel_initializer="uniform"))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                            patience=5, min_lr=0.001)
    model.fit(X_train, y_train,epochs={{choice([25, 50, 75, 100])}},batch_size={{choice([16, 32, 64])}},
              validation_data=(X_val, y_val),callbacks=[reduce_lr])
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())
    X_train, X_val, X_test, y_train, y_val, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save('lstm_model.h5')
