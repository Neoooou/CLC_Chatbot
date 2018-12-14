import unittest
from topicpreprocessing import TopicProcessor
import keras

class TestTopicProcessor(unittest.TestCase):

    def test_init(self):
        tp = TopicProcessor()

    def test__load_files(self):
        tp = TopicProcessor()
        statements, topics = tp._load_files()
        if not len(statements):
            self.fail()

    def test__clean_text(self):
        tp = TopicProcessor()
        statements_1, topics_1 = tp._load_files()
        statements_2, topics_2 = tp._clean_text(statements_1, topics_1)
        self.assertNotEqual(statements_1, statements_2)

    def test__create_tokenizer(self):
        tp = TopicProcessor()
        statements, _ = tp._load_files()
        tokenizer = tp._create_tokenizer(statements)
        if not isinstance(tokenizer, keras.preprocessing.text.Tokenizer):
            self.fail()

    def test__vectorize_text(self):
        tp = TopicProcessor()
        statements, _ = tp._load_files()
        vectorized_text = tp._vectorize_text(statements)
        self.assertNotEqual(statements, vectorized_text)

    def test__pad_lines(self):
        tp = TopicProcessor()
        padded_lines = tp._pad_lines([[40]])
        if not len(padded_lines[0]) > 39 or not 40 in padded_lines[0]:
            self.fail()

    def test__stem_statement(self):
        tp = TopicProcessor()
        original_statement = 'My parents hit me'
        new_statement = tp._stem_statement(original_statement)
        self.assertNotEqual(original_statement, new_statement)

    def test_encode_statement(self):
        tp = TopicProcessor()
        original_statement = 'My parents hit me'
        new_statement = tp.encode_statement(original_statement)
        self.assertNotEqual(original_statement, new_statement)

    def test__build_dictionary(self):
        tp = TopicProcessor()
        _, topics = tp._load_files()
        dictionary, reverse_dictionary, num_topics = tp._build_dictionarys(topics)
        if not len(dictionary) > 1:
            self.fail()

    def test_encode_topic(self):
        tp = TopicProcessor()
        original_topic = 'abuse'
        encoded_topic = tp.encode_topic(original_topic)
        self.assertNotEqual(original_topic, encoded_topic)
        self.assertEqual(encoded_topic, 15)

    def test_decode_topic(self):
        tp = TopicProcessor()
        original_topic = 15
        decoded_topic = tp.decode_topic(original_topic)
        self.assertNotEqual(original_topic, decoded_topic)
        self.assertEqual(decoded_topic, 'abuse')

    def test_get_dataset(self):
        tp = TopicProcessor()
        X, y = tp.get_dataset()
        if not X[0].size > 1:
            self.fail()

    def test_get_num_categories(self):
        tp = TopicProcessor()
        test_num = 4
        tp.num_topics = test_num
        num_topics = tp.get_num_categories()
        self.assertEqual(test_num, num_topics)

    def test_get_sentence_length(self):
        tp = TopicProcessor()
        test_length = 4
        tp.max_sentence_length = test_length
        max_sentence_length = tp.get_sentence_length()
        self.assertEqual(test_length, max_sentence_length)

if __name__ == '__main__':
    unittest.main()
