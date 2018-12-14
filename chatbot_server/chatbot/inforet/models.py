__author__ = 'jaypmorgan'
__version__ = '0.1'
__date__ = '2017-08-19'

import numpy as np 
import nltk 
import re

import spacy
from spacy.language import EntityRecognizer

class InformationRetrieval():
    """
    A class to retrieve the different types of information that 
    will be needed to create a legal case
    """

    def __init__(self):
        self.nlp = spacy.load('en')

    def retrieve_name(self, statement):
        """
        Try and find a name in the statement

        Args:
            statement: the user input

        Returns:
            name: a string of the detected name
            None: if no name is detected then return None
        """
        doc = self.nlp(statement)
        name = []
        for word in doc:
            if word.ent_type_ == 'PERSON':
                name.append(word.text)

        if name:
            name = ' '.join(name)
            return name
        return None
    
    def retrieve_place(self, statement):
        """
        Try and find a place in the statement

        Args:
            statement: the user input

        Returns:
            place: a string of the detected place
            None: if no place is detected then return None
        """
        doc = self.nlp(statement)
        place = []
        place_types = ['ORG', 'GPE', 'FACILITY', 'LOC']
        for word in doc:
            if word.ent_type_ in place_types:
                place.append(word.text)
        
        if place:
            place = ' '.join(place)
            return place
        return None

    def retrieve_time(self, statement): 
        """
        Try and find a time in the statement

        Args:
            statement: the user input

        Returns:
            time: a string of the detected time
            None: if no time is detected then return None
        """
        doc = self.nlp(statement)
        time = []
        time_types = ['TIME', 'DATE', 'ORDINAL', 'CARDINAL']
        for word in doc:
            if word.ent_type_ in time_types:
                time.append(word.text)
        
        if time:
            time = ' '.join(time)
            return time
        return None
    
    def retrieve_email(self, statement):
        """
        Try and find a email in the statement using a REGEX

        Args:
            statement: the user input

        Returns:
            email: a string of the detected email
            None: if no email is detected then return None
        """
        # RFC 5322 email offical standard regex provided by http://emailregex.com/
        regex = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        matches = re.search(regex, statement)
        try:
            email = matches.group(0)
        except:
            email = None
        return email

    def retrieve_phone(self, statement):
        """
        Try and find a phone in the statement using a REGEX

        Args:
            statement: the user input

        Returns:
            phone: a string of the detected phone
            None: if no phone is detected then return None
        """
        regex = r'([0-9]{6,12})'
        phone = re.search(regex, statement)
        try:
            phone = phone.group(0)
        except:
            phone = None
        return phone

if __name__ == '__main__':
    ir = InformationRetrieval()
    name = ir.retrieve_name('My name is John.')
    print(name)
