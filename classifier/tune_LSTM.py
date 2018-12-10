from  keras.wrappers.scikit_learn import  KerasClassifier
from keras.layers import Dense,Activation,Dropout,LSTM,Embedding
from keras.models import Sequential
from keras.optimizers import SGD
from classifier.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.metrics import classification_report
from keras.preprocessing import sequence
def create_model(vocab_size,embedding_vector_length,max_sequence_length,dropout_rate,optimizer):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sequence_length))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='tanh'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())

    model.add(LSTM(100, recurrent_dropout=dropout_rate))
    # model.add(LSTM(units=200,activation='relu'))
    model.add(Dense(units=10, activation="softmax", kernel_initializer="uniform"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
if __name__ == "__main__":
    # preprocess data
    pt = preprocessing("../crawler/clcjsondata.txt")
    docs, labels, y = pt.process()
    y_onehot = to_categorical(y,num_classes=10)
    t = Tokenizer()
    t.fit_on_texts(docs)
    X = t.texts_to_sequences(docs)
    vocab_size = len(t.word_index) + 1
    X_trn,X_tst,y_trn,y_tst = train_test_split(X,y_onehot,random_state=32,test_size=0.2)
    #truncate and pad sequence
    max_sequence_length = 40
    X_trn = sequence.pad_sequences(X_trn, maxlen=max_sequence_length, padding="post")
    X_tst = sequence.pad_sequences(X_tst, maxlen=max_sequence_length, padding="post")
    params = dict(
        dropout_rate=[.5], embedding_vector_length = [400,600],
        max_sequence_length=[max_sequence_length], optimizer = ["rmsprop"],
        vocab_size = [vocab_size], batch_size = [32,64], epochs = [32])
    model = KerasClassifier(build_fn=create_model,verbose = 0)
    grid = GridSearchCV(estimator=model, param_grid=params)
    grid_result = grid.fit(X_trn,y_trn)
    # summarize results
    print("Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_),
          end="\n")
    print("grid scores on development set:")
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]
    for mean,std, params in zip(means,stds,grid.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_tst, grid.predict(X_tst)
    y_pred = to_categorical(y_pred,num_classes=10)
    print(classification_report(y_true, y_pred))
    print()





