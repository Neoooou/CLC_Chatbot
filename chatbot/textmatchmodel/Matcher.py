# author Ran
from keras.models import Sequential,model_from_json
from keras.layers import Dense,Dropout
import os
import numpy as np
from chatbot.textmatchmodel.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import minkowski, cityblock
from keras.layers.advanced_activations import LeakyReLU
from keras import callbacks
import tensorflow as tf
from keras.backend import clear_session
from sklearn.model_selection import KFold


class Matcher:
    vectorizer = TfidfVectorizer(stop_words="english")

    def __init__(self):
        self.docs, self.cleaned_docs, self.y, self.links = preprocessing(self.full_path("clcjsondata.txt")).process()
        self.X = Matcher.vectorizer.fit_transform(self.cleaned_docs)
        # load the model from disk if it was saved
        try:
            self.load_model()
        except (FileNotFoundError, OSError):
            self.model = self.create_model()

            self.train_evaluate_model()

    def full_path(self,filename):
        directory = __file__.split('\\')
        directory = '/'.join(directory[:-1])
        return directory + "/" + filename
    def load_model(self):
        print(self.full_path("model.json"))
        json_file = open(self.full_path("model.json"),'r')

        loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        json_file.close()
        self.model.load_weights(self.full_path("model.h5"))
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        print("loaded text matching model from disk")
    # build the Neural Network model
    def create_model(self):
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units
        # in the first layer, we specify the expected input data shape with vocabulary size
        model.add(Dense(np.power(2, 7), input_dim=self.X.shape[1]))
        model.add(LeakyReLU(.6233))
        model.add(Dropout(.6353))
        model.add(Dense(np.power(2, 5)))
        model.add(LeakyReLU(.5496))
        model.add(Dropout(.6525))
        model.add(Dense(units=10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    # train and evaluate the classification model
    def train_evaluate_model(self):
        y_onehot = to_categorical(self.y, num_classes=10)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                patience=5, min_lr=0.001)
        X_trn, X_tst, y_trn, y_tst = train_test_split(self.X, y_onehot, test_size=0.2, random_state=95)
        X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.2, random_state=95)
        self.model.fit(X_trn, y_trn, epochs=25, batch_size=16, validation_data=(X_val, y_val), callbacks=[reduce_lr])
        scores = self.model.evaluate(X_tst, y_tst)
        self.save_model()
        print("test loss={0}, test accuracy{1}".format(scores[0], scores[1]))
    # save the trained model to the disk
    def save_model(self):
        model_json = self.model.to_json()
        json_file = open("model.json","w")
        json_file.write(model_json)
        self.model.save_weights("model.h5")
        json_file.close()
    # find the nearest paragraph to the input text
    def find_nearest_paragraph(self,text_input,distance = "cosine"):
        # predict the label of input text
        vector_input_w = Matcher.vectorizer.transform(text_input)
        y_onehot = self.model.predict(vector_input_w)
        y_pred = np.argmax(y_onehot)

        # find the nearest paragraph within the range of paragraphs that has predicted label
        # initialize a new vectorizer in the context of matched documents
        v2 = TfidfVectorizer(stop_words="english")
        matched_X_indices = [i for i, e in enumerate(self.y) if e == y_pred]
        matched_docs = [self.docs[i] for i in matched_X_indices]
        matched_docs.extend(text_input)
        matched_X = v2.fit_transform(matched_docs)
        if distance == "cosine":
            # apply cosine similarity to calculate the nearest paragraph
            similarity_metric = cosine_similarity(matched_X[-1], matched_X[:-1])[0].tolist()
            max_similarity_idx = similarity_metric.index(max(similarity_metric))
            matched_indice = matched_X_indices[max_similarity_idx]
        elif distance == "euclidean":
            # apply euclidean distance to calculate the nearest paragraph
            euclidean_metric = euclidean_distances(matched_X[-1], matched_X[:-1])[0].tolist()
            min_distance_idx = euclidean_metric.index(min(euclidean_metric))
            matched_indice = matched_X_indices[min_distance_idx]
        elif distance == "minkowski":
            # apply minkowski distance to calculate the nearest paragraph
            distance_metric = []
            vector_input_m = matched_X[-1]
            for x in matched_X[:-1]:
                d = minkowski(x.todense(), vector_input_m.todense(), p=2)
                distance_metric.append(d)
            min_distance_idx = distance_metric.index(min(distance_metric))
            matched_indice = matched_X_indices[min_distance_idx]
        elif distance == "manhattan":
            # apply Manhattan distance to calculate the nearest paragraph
            distance_metric = []
            vector_input_m = matched_X[-1]
            for x in matched_X[:-1]:
                d = cityblock(x.todense(), vector_input_m.todense())
                distance_metric.append(d)
            min_distance_idx  = distance_metric.index(min(distance_metric))
            matched_indice = matched_X_indices[min_distance_idx]
        return self.links[matched_indice], self.docs[matched_indice]

    def reweight(self):
        '''find the appropriate weights for each word in input message while calculating distance,
         which means that the more relevant between a word in input message and the classified label,
        the more important it should be'''
        #unknown
        pass

    def k_fold_test(self):
        accs = []
        losses = []
        one_hot_y = to_categorical(self.y ,num_classes=10)
        kf = KFold(n_splits=10)
        for train, test in kf.split(self.X):
            model = self.create_model()
            model.fit(self.X[train], one_hot_y[train],  epochs=25, batch_size=16)
            score = model.evaluate(self.X[test], one_hot_y[test])
            accs.append(score[1])
            losses.append(score[0])
        print("Accuracy: %.2f%% (+/- %.2f%% )" % (np.mean(accs), np.std(accs)))
        print("Loss: %.2f%% (+/- %.2f%%)" % (np.mean(losses), np.std(losses)))
if __name__ == "__main__":
    matcher = Matcher()
    #text_input = ["Hi hi hi hi I was bullied"]
    #matcher.find_nearest_paragraph(text_input,distance="cosine")
    matcher.k_fold_test()




