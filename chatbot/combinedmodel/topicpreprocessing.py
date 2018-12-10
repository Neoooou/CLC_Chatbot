__author__ = 'jaypmorgan'
__version__ = '0.1'
__date__ = '2017-08-19'

import re
import numpy as np

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

class TopicProcessor():

    def __init__(self, corpus_dir='chatbot/corpus/', corpus_files=['topics.txt']):
        self.max_sentence_length = 10
        self.tokenizer = Tokenizer()
        self.corpus_dir = corpus_dir
        self.corpus_files = corpus_files
        self.stemmer = PorterStemmer()

        statements, topics = self._load_files()
        self.tokenizer = self._create_tokenizer(statements)
        self.statements, self.topics = self._clean_text(statements, topics)
        self.dictionary, self.reverse_dictionary, self.num_topics = self._build_dictionarys(self.topics)

    def _load_files(self):
        regex = r'(?:\[(.*)\])?\s?(.*)'
        matcher = re.compile(regex, re.IGNORECASE)
        topics = []
        statements = []
        for file_name in self.corpus_files:
            with open(str(self.corpus_dir + file_name), 'r') as open_file:
                for line in open_file:
                    matches = matcher.search(line)
                    try:
                        topics.append(matches.group(1))
                    except:
                        topics.append('none')
                    statement = matches.group(2)
                    statement = statement.replace('{answer}', '')
                    statement = self._stem_statement(statement)
                    statements.append(statement)
                    if len(statement) > self.max_sentence_length:
                        self.max_sentence_length = len(statement)
        return statements, topics
            
    def _clean_text(self, X, y):
        X = self._vectorize_text(X)
        return X, y

    def _create_tokenizer(self, statements):
        self.tokenizer.fit_on_texts(statements)
        return self.tokenizer

    def _vectorize_text(self, statements):
        vectorized_text = self.tokenizer.texts_to_sequences(statements)
        vectorized_text = self._pad_lines(vectorized_text)
        return vectorized_text

    def _pad_lines(self, lines):
        padded_lines = pad_sequences(lines, maxlen=self.max_sentence_length)
        return padded_lines

    def _stem_statement(self, statement):
        statement = [word.lower() for word in text_to_word_sequence(statement) if word not in stopwords.words('english')]
        statement = [self.stemmer.stem(word) for word in statement]
        statement = " ".join(statement)
        return statement

    def encode_statement(self, statement):
        statement = self._stem_statement(statement)
        vectorized_statement = self.tokenizer.texts_to_sequences([statement])
        vectorized_statement = self._pad_lines(vectorized_statement)
        return np.array(vectorized_statement)

    def _build_dictionarys(self, topics):
        num_topics = len(set(topics))
        dictionary = {}
        reverse_dictionary = {}
        counter = 0
        for topic in topics:
            if topic not in dictionary.keys():
                dictionary[topic] = counter
                reverse_dictionary[counter] = topic
                counter += 1

        return dictionary, reverse_dictionary, num_topics

    def encode_topic(self, topic):
        topic_id = self.dictionary[topic]
        return topic_id

    def decode_topic(self, topic_id):
        topic = self.reverse_dictionary[topic_id]
        return topic
    
    def get_dataset(self):
        X = np.reshape(self.statements, newshape=(len(self.statements), self.max_sentence_length))
        y = np.array([self.encode_topic(topic) for topic in self.topics])
        return X, y

    def get_num_categories(self):
        return self.num_topics

    def get_sentence_length(self):
        return self.max_sentence_length

if __name__ == '__main__':
    td = TopicProcessor()
    X, y = td.get_dataset()
    print(td.num_topics)
    print(td.max_sentence_length)