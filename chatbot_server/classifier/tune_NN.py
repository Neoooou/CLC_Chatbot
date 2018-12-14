from  keras.wrappers.scikit_learn import  KerasClassifier
from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from classifier.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
def create_model(dropout_rate,optimizer,activation_):
    model = Sequential()
    model.add(Dense(units=512,
                    input_dim= 2434, activation=activation_))  # input layer
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation=activation_))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=10,activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
if __name__ == "__main__":
    # preprocess data
    pt = preprocessing("../crawler/clcjsondata.txt")
    docs, y, links = pt.process()
    y_onehot = to_categorical(y,num_classes=10)
    v = TfidfVectorizer(stop_words="english")
    # convert paragraphs into vectors
    X = v.fit_transform(docs).toarray()
    X_trn,X_tst,y_trn,y_tst = train_test_split(X,y_onehot,random_state=32,test_size=0.2)
    vocab_size = len(X[0])
    # record best optimizer here   hyperas
    params = dict(
        activation_ = ["relu"], dropout_rate=[ .5,.7],epochs = [32],
        optimizer = ["rmsprop"], batch_size = [32,64])
    model = KerasClassifier(build_fn=create_model,verbose = 1)
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