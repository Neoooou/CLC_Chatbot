from classifier.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.text import one_hot,Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# pre-process data
docs,y,links = preprocessing("../crawler/clcjsondata.txt").process()
# integer encode documents
t = Tokenizer()
t.fit_on_texts(docs)
X = t.texts_to_sequences(docs)
one_hot_y = to_categorical(y,num_classes=10)
X_trn,X_tst,y_trn,y_tst = train_test_split(X,one_hot_y,test_size=0.2,random_state=32)

# truncate and pad input sequences
max_sequence_length = 18
X_trn = sequence.pad_sequences(X_trn,maxlen=max_sequence_length,padding="post")
X_tst = sequence.pad_sequences(X_tst,maxlen=max_sequence_length,padding="post")
print(X_trn.shape)
vocab_size = len(t.word_index) + 1
#instantiate model
model = Sequential()
embedding_vector_length = 500
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sequence_length))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='tanh'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=10, activation="softmax", kernel_initializer="uniform"))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
model.fit(X_trn,y_trn, validation_split=0.1,epochs=32,batch_size=48)
# evaluate
y_pred = model.predict(X_tst)
scores = model.evaluate(X_tst,y_tst,verbose=1)
print("evaluate score", scores)


