import re
from pprint import pprint
from datetime import datetime

#from .report import PDFGen
from .inforet.models import InformationRetrieval
from .combinedmodel.models import Conversation
from .textmatchmodel.Matcher import Matcher


class ChatBot:
    # Requirement to simulate required details in the case management DB.
    topic_requirements = [{
        'topic_name': 'abuse',
        'required': [
            'name',
            'where',
            'time',
            'email',
            'phone',
            'contactTime',
        ]
    }, {
        'topic_name': 'cyber-crime',
        'required': [
            'name',
            'email',
            'phone',
        ]
    }, {
        'topic_name': 'UNK',
        'required': [
            'name',
            'email',
            'phone',
        ]
    }]

    INDEX = 0
    QUESTION_ASKED = ''
    # if pivotal statement changed, call match model to give the most relevant link and doc to it
    pivotal_statement_switch = False

    def __init__(self):
        self.recreate()
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.infoRetrieval = InformationRetrieval()
        self.convModel = Conversation()
        self.conversation['datetime-started'] = datetime.now().strftime(self.date_format)
        self.matcher = Matcher()

    def recreate(self):
        self.conversation = {
            'datetime-started': None,
            'datetime-finished': None,
            'conversation-length': None,
            'pivotal-statement': None,
            'advisor': False,
            'rights': False,
            'context': [],
            'topic': {
                'topic_name': None,
                'confidence': 0
            },
            'required': [],
            'retrieve_details': False,
        }

    def render_report(self):
        """
        Once the conversation has been concluded, the PDF report can
        be generated. This example uses the datetime as the title of the
        report
        """
        self.conversation['datetime-finished'] = datetime.now().strftime(self.date_format)

        start = datetime.strptime(self.conversation['datetime-started'], self.date_format)
        end = datetime.strptime(self.conversation['datetime-finished'], self.date_format)
        self.conversation['conversation-length'] = (end - start).total_seconds() / 60
        '''
        gen = PDFGen()
        gen.create_pdf(
            details=self.conversation['required'],
            topic_name=self.conversation['topic']['topic_name'],
            topic_confidence=self.conversation['topic']['confidence'],
            conversation_started=self.conversation['datetime-started'],
            conversation_finished=self.conversation['datetime-finished'],
            context=self.conversation['context'],
            pivotal_statement=self.conversation['pivotal-statement'],
            advisor=self.conversation['advisor'],
            rights=self.conversation['rights'],
            length=self.conversation['conversation-length']
        )'''

    def request(self, statement):
        """
        Main request function to interact with the chatbot, simply supply
        the users input and you will receive the response.

        args:
            statement: the users statement

        returns:
            response: the response to the users input
            conversation: internal JSON of the conversations context
            parse_user_statement: a HTML friendly version of the input statement
        """
        if self.conversation['datetime-started'] is None:
            self.conversation['datetime-started'] = datetime.now().strftime(self.date_format)
        self.conversation['context'].append('[User]: ' + statement)
        self.retrieve_name(statement)

        # scan for the details, make sure to get an answer if the bot has asked for ir
        asking_place = True if self.QUESTION_ASKED == 'where' else False
        self.retrieve_place(statement, asking=asking_place)
        asking_time = True if self.QUESTION_ASKED == 'time' else False
        self.retrieve_time(statement, asking=asking_time)
        asking_email = True if self.QUESTION_ASKED == 'email' else False
        self.retrieve_email(statement, asking=asking_email)
        asking_contact_time = True if self.QUESTION_ASKED == 'contactTime' else False
        if asking_contact_time:
            self.retrieve_contact_time(statement, asking=True)
        self.retrieve_phone(statement)

        # calculate the response from the Conversation model
        if self.conversation['retrieve_details'] is False:
            if 'context_set' in self.conversation and self.conversation['context_set'] == 'learnMore':
                response, context_set, pred_class, topic, confidence = self.convModel.request(statement,
                                                                              self.conversation['context_set'],
                                                                              'learnMore')
            elif 'context_set' in self.conversation:
                response, context_set, pred_class, topic, confidence= self.convModel.request(statement,
                                                                              self.conversation['context_set'])
            else:
                response, context_set, pred_class, topic, confidence = self.convModel.request(statement)
        if pred_class == 'statement':
            print("statement：",statement)
            self.retrieve_topic(topic, confidence)
            self.conversation['pivotal-statement'] = statement
            self.pivotal_statement_switch = True

        # set flag in the conversation for the PDF report to act on
        try:
            if context_set is not None:
                self.conversation['context_set'] = context_set
                if context_set == 'retrieveDetails':
                    self.conversation['retrieve_details'] = True
        except:
            pass

        # set the advisor flag, then ask for the details required for the case
        if self.conversation['retrieve_details'] is True:
            self.conversation['advisor'] = True
            try:
                # iterate through the list of answers that haven't been gained yet
                while list(self.conversation['required'][self.INDEX].values())[0] is not None:
                    self.INDEX += 1
            except IndexError:
                # we've come to the end of the list, finally ask what time to be contacted
                response, _ = self.convModel.request(statement, 'contactTime', 'contactTime')
                return response, self.conversation, self.parse_user_statement(statement)
            # ask the question from the list
            if list(self.conversation['required'][self.INDEX].values())[0] is None:
                context_required = list(self.conversation['required'][self.INDEX])[0]
                self.QUESTION_ASKED = context_required
                response, context_set, pred_class, _, _ = self.convModel.request(statement, context_required,
                                                                               'detailsRetrieve')
        # inject the name
        if re.search(r'{name}', response):
            for index, details in enumerate(self.conversation['required']):
                if 'name' in details:
                    if self.conversation['required'][index]['name'] is not None:
                        response = response.replace('{name}', self.conversation['required'][index]['name'])

        # inject the html link back to the CLC website, would pull from a database
        if '{topic-link}' in response:
            self.conversation['rights'] = True
            response = response.replace('{topic-link}',
                                        str('{' + self.conversation['topic']['topic_name'] + '-topic-link}'))

        response = response.replace('{name}', '').replace(' ,', ',')
        self.conversation['context'].append('[Chatbot]: ' + response)
        doc = None
        link = None
        if self.conversation["pivotal-statement"] is not None and self.pivotal_statement_switch:
            link, doc = self.matcher.find_nearest_paragraph([self.conversation["pivotal-statement"]])
            response = response + " [Recommended webpage is shown]"
            doc = 'Recommended paragraph:\n '+doc
            self.pivotal_statement_switch = False
        return response, doc, link

    def parse_user_statement(self, statement):
        """
        If the user has said something in the statement that the chatbot has used
        to answer a question, this returns a HTML friendly snippet that demonstrates
        that fact

        args:
            statement: the users statement

        returns:
            statement: HTML friendly statement
        """
        # iterate through the answers have been saved
        for index, value in enumerate(self.conversation["required"]):
            pattern = str(list(value.values())[0])
            if pattern != 'None':
                replacement_string = " " + pattern
                statement = re.sub(pattern, str(
                    '<span id="marker">' + replacement_string + '<div id="tooltip">I\'ve learnt something from this.</div></span>'),
                                   statement)
        return statement

    def retrieve_contact_time(self, statement, asking=False):
        """
        Scan the user statement for a time

        args:
            statement: the user statement
            asking: whether the chatbot has asked for the information.
            If True, then this will add the whole statement to the JSON
            array if no time can be found

        returns:
            None: if nothing is found in the statement and asking is false
        """
        contact_time = self.infoRetrieval.retrieve_place(statement)
        if asking is True and contact_time is None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'contactTime' in required_details:
                    self.conversation['required'][index]['contactTime'] = statement
        if contact_time is not None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'contactTime' in required_details:
                    self.conversation['required'][index]['contactTime'] = contact_time
                    return None
            self.conversation['required'].append({'contactTime': contact_time})

    def retrieve_place(self, statement, asking=False):
        """
        Scan the user statement for a place

        args:
            statement: the user statement
            asking: whether the chatbot has asked for the information.
            If True, then this will add the whole statement to the JSON
            array if no place can be found

        returns:
            None: if nothing is found in the statement and asking is false
        """
        place = self.infoRetrieval.retrieve_place(statement)
        if asking is True and place is None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'where' in required_details:
                    self.conversation['required'][index]['where'] = statement
        if place is not None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'where' in required_details:
                    self.conversation['required'][index]['where'] = place
                    return None
            self.conversation['required'].append({'where': place})

    def retrieve_name(self, statement):
        """
        Scan the user statement for a name

        args:
            statement: the user statement
            asking: whether the chatbot has asked for the information.
            If True, then this will add the whole statement to the JSON
            array if no name can be found

        returns:
            None: if nothing is found in the statement and asking is false
        """
        name = self.infoRetrieval.retrieve_name(statement)
        if name is not None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'name' in required_details:
                    self.conversation['required'][index]['name'] = name
                    return None
            self.conversation['required'].append({'name': name})

    def retrieve_time(self, statement, asking=False):
        """
        Scan the user statement for a time

        args:
            statement: the user statement
            asking: whether the chatbot has asked for the information.
            If True, then this will add the whole statement to the JSON
            array if no time can be found

        returns:
            None: if nothing is found in the statement and asking is false
        """
        time = self.infoRetrieval.retrieve_time(statement)
        if asking is True and time is None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'time' in required_details:
                    self.conversation['required'][index]['time'] = statement
        if time is not None:
            for index, required_details in enumerate(self.conversation['required']):
                if 'time' in required_details:
                    self.conversation['required'][index]['time'] = time
                    return None
            self.conversation['required'].append({'time': time})

    def retrieve_email(self, statement, asking=True):
        """
        Scan the user statement for a email

        args:
            statement: the user statement
            asking: whether the chatbot has asked for the information.
            If True, then this will add the whole statement to the JSON
            array if no email can be found

        returns:
            None: if nothing is found in the statement and asking is false
        """
        email = self.infoRetrieval.retrieve_email(statement)
        if asking is True and email is None:
            for index, details in enumerate(self.conversation['required']):
                if 'email' in details:
                    self.conversation['required'][index]['email'] = email
        if email is not None:
            for index, details in enumerate(self.conversation['required']):
                if 'email' in details:
                    self.conversation['required'][index]['email'] = email
                    return None
            self.conversation['required'].append({'email': email})

    def retrieve_phone(self, statement):
        """
        Scan the user statement for a phone number

        args:
            statement: the user statement
            asking: whether the chatbot has asked for the information.
            If True, then this will add the whole statement to the JSON
            array if no number can be found

        returns:
            None: if nothing is found in the statement and asking is false
        """
        phone = self.infoRetrieval.retrieve_phone(statement)
        if phone:
            for index, value in enumerate(self.conversation['required']):
                if 'phone' in value:
                    self.conversation['required'][index]['phone'] = phone

    def retrieve_topic(self, topic, confidence):
        """
        Determine whether a conversation can be gained from the combined model.
        If the confidence is above a threshold value, then add that to the conversation

        args:
            topic: the topic type returned from the combined model
            confidence: float value of the confidence for the topic returned by the
            combined model.
        """
        if confidence >= 0.5 and self.conversation['topic']['topic_name'] is None:
            self.conversation['topic']['topic_name'] = topic
            self.conversation['topic']['confidence'] = float(confidence)
            self.load_requirements(topic)
        elif confidence < 0.5 and self.conversation['topic']['topic_name'] is None:
            self.load_requirements('UNK')

    def load_requirements(self, topic):
        """
        Load the JSON requirements given that we now know what type of topic
        we are dealing with.

        args:
            topic: the legal case type we are dealing with.
        """
        for topic_name in self.topic_requirements:
            if topic_name['topic_name'] == topic:
                required_details = self.conversation['required']
                details = []
                for detail in required_details:
                    details.append(list(detail.keys())[0])
                for detail in topic_name['required']:
                    if detail not in details:
                        self.conversation['required'].append({detail: None})


if __name__ == '__main__':
    cb = ChatBot()
    print('Hi, how might I help you today?')
    while True:
        user_input = input('> ')
        if user_input == "":
            print('Thank you for talking with me today.')
            print('\nHere is a full transcript of the conversation: ')
            print(cb.conversation)
            break
        else:
            print(cb.request(user_input))
