__author__ = 'jaypmorgan'
__version__ = '0.1'
__date__ = '2017-08-19'

import json
import numpy as np 

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class ConversationProcessor():
    MAX_SEQUENCE_LENGTH = 48
    STEMMER = PorterStemmer()

    def __init__(self):
        X, y, classes, n_classes = self._load_intents()
        self.tokenizer = Tokenizer()
        self.tokenizer = self._create_tokenizer(X)
        self.X = self._encode_statements(X)
        self.y = y
        self.classes = classes
        self.n_classes = n_classes

    def get_class(self, index):
        return self.classes[index]
        
    def encode_statment(self, statement):
        encoded_statement = self._clean_text(statement)
        encoded_statement = self.tokenizer.texts_to_sequences([encoded_statement])
        encoded_statement = self._pad_lines(encoded_statement)
        return np.array(encoded_statement)

    def get_dataset(self):
        return np.array(self.X), np.array(self.y)
    
    def _encode_statements(self, statements):
        encoded_statements = self.tokenizer.texts_to_sequences(statements)
        encoded_statements = self._pad_lines(encoded_statements)
        return encoded_statements    

    def _pad_lines(self, lines):
        padded_lines = pad_sequences(lines, maxlen=self.MAX_SEQUENCE_LENGTH)
        return padded_lines
    
    def _create_tokenizer(self, statements):
        self.tokenizer.fit_on_texts(statements)
        return self.tokenizer

    def _clean_text(self, statement):
        statement = [word.lower() for word in nltk.word_tokenize(statement) if word not in stopwords.words('english')]
        statement = [self.STEMMER.stem(word) for word in statement]
        statement = " ".join(statement)
        return statement

    def _load_intents(self):
        x_y = []
        classes = []

        file_location = 'chatbot/corpus/output.json'
        with open(file_location, 'r') as open_file:
            json_file = json.load(open_file)

        for intent in json_file['intents']:
            if intent['tag'] not in classes:
                classes.append(intent['tag']) 
            for pattern in intent['patterns']:
                statement = self._clean_text(pattern)
                x_y.append((statement, intent['tag']))

        n_classes = len(classes)
        y_template = [0] * n_classes
        y = []
        X = []
        for row in x_y:
            X.append(row[0])
            y_tmp = list(y_template)
            y_tmp[classes.index(row[1])] = 1
            y.append(y_tmp)
        
        return X, y, classes, n_classes

if __name__ == '__main__':
    cp = ConversationProcessor()
