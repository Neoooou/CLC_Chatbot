import os
import unittest
from models import Conversation
from convpreprocessing import ConversationProcessor
from topicpreprocessing import TopicProcessor

class TestConversation(unittest.TestCase):
    
    def test_init(self):
        convModel = Conversation()

    def test_init_with_intents(self):
        convModel = Conversation('interface/chat/corpus/output.json')

    def test__load_model(self):
        model = Conversation._load_model(Conversation)
        if not model:
            self.fail('no model returned from function')

    def test__save_model(self):
        model = Conversation._load_model(Conversation)
        Conversation._save_model(Conversation, model)
        if not os.path.isfile(
            'interface/chat/chatbot/combinedmodel/trainedconversationalmodelweights.h5'
        ):
            self.fail('unable to find saved file')

    def test_encode(self):
        convModel = Conversation()
        encoded_statement = convModel.encode('My parents is a test')
        if 7 not in encoded_statement:
            self.fail('encoded statement is unexpected')

    def test_request(self):
        convModel = Conversation()
        response, context, classes, topic, conf = convModel.request('Hello my name is Jay')
        if 'Hello {name}' not in response and classes is not 'greetings':
            self.fail('didnt expect that output')

    # should only be run if you machine is powerful enough
    # else omit
    # def test_init_with_testing(self):
    #     convModel = Conversation(testing=True)

    # def test__build_model(self):
    #     model = Conversation._build_model(Conversation)
    #     if not model:
    #         self.fail('no model returned from function')

    # def test__test_model(self):
    #     convModel = Conversation(testing=True)

    # def test__train_model(self):
    #     td = TopicProcessor()
    #     cp = ConversationProcessor()
    #     convX, convy = cp.get_dataset()

    #     topX, topy = td.get_dataset()
    #     topy = np_utils.to_categorical(topy)
    #     top_classes = td.get_num_categories()
    #     top_sentence_length = td.get_sentence_length()

    #     model = Conversation._build_model(Conversation)
    #     model = Conversation._train_model(
    #         Conversation,
    #         model,
    #         convX,
    #         convy,
    #         topX,
    #         topy
    #     )

    #     if not model:
    #         self.fail()

if __name__ == '__main__':
    unittest.main()
