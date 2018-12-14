import os
import unittest
from convpreprocessing import ConversationProcessor

class TestConversationProcessor(unittest.TestCase):

    def test_init(self):
        convpro = ConversationProcessor()
        if not convpro.n_classes:
            self.fail()

    def test_get_class(self):
        convpro = ConversationProcessor()
        if convpro.get_class(0) != 'statement':
            self.fail()

    def test_encode_statement(self):
        convpro = ConversationProcessor()
        statement = convpro.encode_statment('My parents hit me')
        if not statement.any():
            self.fail('values in encoded array was not as expected')

    def test_get_dataset(self):
        convpro = ConversationProcessor()
        X, y = convpro.get_dataset()
        if X[0].size is not 48:
            self.fail()

    def test__encode_statements(self):
        convpro = ConversationProcessor()
        statements = convpro._encode_statements([
            'My parents hit me',
            'My parents were mean to me',
        ])
        if not statements[0].any() and not statements[1].any():
            self.fail()

    def test__pad_lines(self):
        convpro = ConversationProcessor()
        padded_line = convpro._pad_lines([[41]])
        if padded_line[0].size is not 48:
            self.fail()

    def test__clean_text(self):
        convpro = ConversationProcessor()
        expected = 'my parent hit'
        statement = convpro._clean_text('My parents hit me')
        self.assertEqual(expected, statement)

    def test__load_intents(self):
        convpro = ConversationProcessor()
        X, y, classes, n_classes = convpro._load_intents()
        if not len(X):
            self.fail()

if __name__ == '__main__':
    unittest.main()