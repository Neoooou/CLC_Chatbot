"""
Conversational model that takes in user input and converses with the
front end.
"""


import json
import random
import numpy as np
from sys import platform
if platform != 'darwin':
    import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.utils import np_utils
from keras.models import model_from_json
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
import keras.backend as K

from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os
import tensorflow as tf
from keras.backend import clear_session
from chatbot.seq2seq.seq2seqmodel import seq2seqmodel
try:
    from convpreprocessing import ConversationProcessor
    from topicpreprocessing import TopicProcessor
except ImportError:
    from .convpreprocessing import ConversationProcessor
    from .topicpreprocessing import TopicProcessor


class Conversation():
    """ 
    This model will learn from user input to bot output.
    Given the probability of the user's statement being of a certain
    type, the result will be a random item in the response list.
    """

    def __init__(self, intents=None, testing=False):
        self.td = TopicProcessor()
        self.cp = ConversationProcessor()
        self.convX, self.convy = self.cp.get_dataset()

        self.topX, self.topy = self.td.get_dataset()
        self.topy = np_utils.to_categorical(self.topy)
        self.top_classes = self.td.get_num_categories()
        self.top_sentence_length = self.td.get_sentence_length()
        self.graph = None
        self.seq2seqmodel = seq2seqmodel()
        if not testing:
            # Try to load an existing model from file
            try:
                clear_session()
                self.model, self.graph, self.sess = self._load_model()
            except (FileNotFoundError, OSError):
                # If the model doesn't exist, we'll have to build one
                model = self._build_model()
                self.model = self._train_model(model, self.convX, self.convy, self.topX, self.topy)
                self.model, self.graph, self.sess = self._load_model()
            # If the user hasn't supplied a location, use the default
            if not intents:
                intents = 'chatbot/corpus/output.json'

            with open(intents, 'r') as json_file:
                self.intents = json.load(json_file)
        else:
            # self._test_model(self.convX, self.convy, self.topX, self.topy)
            self._k_fold_test(self.convX, self.convy, self.topX, self.topy)
            

    def _load_model(self):
        """
        Load the model from the disk. This is the weights in an h5 file,
        and the structure in a JSON format.

        returns: keras model
        """
        directory = __file__.split('\\')
        directory = '/'.join(directory[:-1])
        #load_model = models.load_model(os.path.join(directory, 'model.h5'))
        with open(os.path.join(directory, 'trainedconversationalmodel-old.json'), 'r') as open_file:
            load_model = open_file.read()
        load_model = model_from_json(load_model)
        load_model.load_weights(os.path.join(directory, 'trainedconversationalmodelweights-old.h5'))
        load_model._make_predict_function()
        self.graph = tf.get_default_graph()
        # self.graph.finalize()
        sess = K.get_session()
        print("Combined model was loaded from disk")
        return load_model, self.graph, sess

    def _save_model(self, model):
        """
        Save a trained model to the disk. The weights of the model is saved in
        a H5 file. The models structure is in a JSON file format.

        Args:
            model: The trained model you want to save to disk.
        """

        directory = __file__.split('\\')
        directory = '/'.join(directory[:-1])
        model.save(os.path.join(directory, 'model.h5'))
        json_model = model.to_json()
        with open(directory + '/trainedconversationalmodel.json', 'w') as open_file:
            open_file.write(json_model)
        model.save_weights(directory + '/trainedconversationalmodelweights.h5')

    def _build_model(self):
        """
        Build keras model and train it against the dataset.

        Args:
            X: input statements
            y: label of the input statement

        Returns:
            A trained keras model
        """
        conv_input = Input(shape=(self.cp.MAX_SEQUENCE_LENGTH,), name='conv_input')
        top_input = Input(shape=(self.top_sentence_length,), name='top_input')
        inputs = concatenate([conv_input, top_input])
        embedding = Embedding(input_dim=1000, output_dim=40)(inputs)
        flatten = Flatten()(embedding)

        conv_layer_1 = Dense(units=600, activation='sigmoid')(flatten)
        conv_dropout_1 = Dropout(0.7)(conv_layer_1)
        conv_layer_2 = Dense(units=400, activation='sigmoid')(conv_dropout_1)
        conv_dropout_2 = Dropout(0.7)(conv_layer_2)
        conv_output = Dense(units=self.cp.n_classes, activation='softmax', name='conversation_output')(conv_dropout_2)

        topic_layer_1 = Dense(units=2000, activation='sigmoid', init='normal')(flatten)
        topic_dropout_1 = Dropout(0.5)(topic_layer_1)
        topic_layer_2 = Dense(units=500, activation='sigmoid', init='uniform')(topic_dropout_1)
        topic_dropout_2 = Dropout(0.5)(topic_layer_2)
        topic_output = Dense(units=self.top_classes, activation='softmax', name='topic_output')(topic_dropout_2)
        topic_optimizer = optimizers.RMSprop(lr=0.0001)

        model = Model(inputs=[conv_input, top_input], outputs=[conv_output, topic_output])
        model.compile(optimizer=topic_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _k_fold_test(self, statements, labels, sentences, topics):
        """
        Perform Kfold cross validation, this is a slightly different take on the _test_model function, as
        the X and Y datasets will be split into k partitions and then tests k times. This means that it will
        take considerably longer to get the final result, depending on the number of folds used.

        args:
            statements: the intent statements (X)
            labels: the intent classes (Y)
            sentences: the topic statements (X)
            topics: the topic classes (Y)
        """
        cv_scores_conv = []
        cv_scores_topic = []
        kfold = KFold(labels.shape[0], n_folds=10, shuffle=True, random_state=0) # depreciated in sklearn, but works well with multi-label classification
        for train, test in kfold:
            model = self._build_model()
            model.fit([statements[train], sentences[train]], [labels[train], topics[train]], batch_size=742, epochs=1300)
            score = model.evaluate([statements[test], sentences[test]], [labels[test], topics[test]])
            print(score[3] * 100, score[4] * 100)
            cv_scores_conv.append(score[3] * 100)
            cv_scores_topic.append(score[4] * 100)

        print("Intent Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores_conv), np.std(cv_scores_conv)))
        print("Topic Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores_topic), np.std(cv_scores_topic)))


    def _test_model(self, statements, labels, sentences, topics):
        """
        Perform a simple testing method by splitting the corpus into training and testing sets. This method
        will also output graphs for how the model performs during the training and testing methods. 

        I have disabled it it on MacOS as matplotlib doesn't work in virtualenvwrapper environments.

        args:
            statements: the intent statements (X)
            labels: the intent classes (Y)
            sentences: the topic statements (X)
            topics: the topic classes (Y)
        """
        X_conv_train, X_conv_test, y_conv_train, y_conv_test = train_test_split(statements, labels, test_size=0.5, random_state=0)
        X_topic_train, X_topic_test, y_topic_train, y_topic_test = train_test_split(sentences, topics, test_size=0.5, random_state=0)

        images_folder_loc = 'images/'
        model = self._build_model()
        training = model.fit([X_conv_train, X_topic_train], [y_conv_train, y_topic_train], validation_data=([X_conv_test, X_topic_test], [y_conv_test, y_topic_test]), batch_size=X_conv_train.shape[0], epochs=1500)
        print(training.history.keys())

        if platform != 'darwin':
            plt.plot(training.history['conversation_output_acc'], color='red')
            plt.plot(training.history['val_conversation_output_acc'], color='green')
            plt.title('Conversation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(str(images_folder_loc + 'conv_acc.png'))
            plt.clf()

            plt.plot(training.history['topic_output_acc'], color='red')
            plt.plot(training.history['val_topic_output_acc'], color='green')
            plt.title('Topic Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(['Training', 'Testing'], loc='upper left')
            plt.savefig(str(images_folder_loc + 'topic_acc.png'))
            plt.clf()

            plt.plot(training.history['conversation_output_loss'], color='red')
            plt.plot(training.history['val_conversation_output_loss'], color='green')
            plt.title('Converation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(['Training', 'Testing'], loc='upper left')
            plt.savefig(str(images_folder_loc + 'conv_loss.png'))
            plt.clf()

            plt.plot(training.history['topic_output_loss'], color='red')
            plt.plot(training.history['val_topic_output_loss'], color='green')
            plt.title('Topic Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(['Training', 'Testing'], loc='upper left')
            plt.savefig(str(images_folder_loc + 'topic_loss.png'))
            plt.clf()
        
        plot_model(model, to_file=str(images_folder_loc + 'model.png'))
        
        print(model.metrics_names)
        print(model.evaluate([X_conv_test, X_topic_test], [y_conv_test, y_topic_test]))

    def _train_model(self, model, statements, labels, sentences, topics):
        """
        Train method for the pre-compiled model. Returns a trained model

        args:
            model: pre-compiled keras model to train
            statements: the intent statements (X)
            labels: the intent classes (Y)
            sentences: the topic statements (X)
            topics: the topic classes (Y)

        returns:
            model: the trained keras model
        """
        model.fit([statements, sentences], [labels, topics], batch_size=statements.shape[0], epochs=1)
        self._save_model(model)
        return model

    def encode(self, statement):
        """
        When the user inputs a statement into the chatbot. It will need to be converted
        to a sequence vector for the model to work with. This function is an abstraction of that
        process.

        Args:
            statement: statement that is to be converted to a vector.

        Returns:
            A sequence vector representing the user's input.
        """
        return self.cp.encode_statment(statement)

    def request(self, req, context=None, state=None):
        """
        This is the function that will be called by the chatbot, when the user inputs a statement.
        It will call the encode function of this class, use the model to predict the 
        probability that it is of all classes using a softmax function. The highest probability 
        is therefore deemed the correct one (with some error threshold checking).

        Args:
            req: statement that we need to predict the class for.

        Returns:
            (response, context_set). Response is the String that the chatbot should respond with.
            Context_set is the tag of the intent that the bot needs context for.
        """
        with self.sess.as_default():
            with self.graph.as_default():
                pred = self.model.predict([self.encode(req), self.td.encode_statement(req)])
        statement_pred = pred[0]
        topic_pred = pred[1]
        topic_class = np.argmax(topic_pred)
        topic_confidence = topic_pred[0][topic_class]

        statement_class = np.argmax(statement_pred)
        statement_confidence = statement_pred[0][statement_class]
        
        pred_class = self.cp.get_class(statement_class)
        topic_pred_class = self.td.decode_topic(topic_class)
        context_set = None
        response = ''

        for intent in self.intents['intents']:
            if context:
                if intent['tag'] == state:
                    for index, value in enumerate(intent['context_filter']):
                        if value == context:
                            if 'context_set' in intent:
                                try:
                                    context_set = intent['context_set'][index]
                                except:
                                    context_set = None
                            response = intent['responses'][index]
                elif 'context_filter' in intent and intent['tag'] == pred_class:
                    for index, value in enumerate(intent['context_filter']):
                        if value == context:
                            if 'context_set' in intent:
                                try:
                                    context_set = intent['context_set'][index]
                                except:
                                    context_set = None
                            response = intent['responses'][index]
            elif intent['tag'] == pred_class and 'context_filter' not in intent:
                if 'context_set' in intent:
                    context_set = intent['context_set'][0]
                response = random.choice(intent['responses'])

        if not response: # catch all
#            response = 'I am not sure I understand, could you explain it to me further?'
            response = self.seq2seqmodel.decode_sequence(req)

        return response, context_set, pred_class, topic_pred_class, topic_confidence

if __name__ == '__main__':
    cb = Conversation(testing=True)
