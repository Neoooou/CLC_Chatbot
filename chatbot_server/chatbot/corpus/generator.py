import re
import os
import json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

json_output = {
    'intents': [{
        'tag': 'statement',
        'patterns': [],
        'responses': ['I am sorry to hear about that. Would you like to tell me more about that situation?'],
        'context_set': ['learnMore']
    }, {
        'tag': 'greetings',
        'patterns': [],
        'responses': ['Hello {name}, how may I help you today?']
    }, {   
        'tag': 'positiveReq',
        'patterns': [],
        'responses': ['You can find them here: {topic-link}. Would you like to speak with one of our advisors?', 'Okay, I am just going to ask a couple of questions first. Is that okay?', ''],
        'context_filter': ['infoReq', 'advisor', 'advisorPostive'],
        'context_set': ['advisor', 'advisorPostive', 'retrieveDetails']
    }, {
        'tag': 'negativeReq',
        'patterns': [],
        'responses': ["That's okay, would you prefer to speak with someone instead?", 'Okay. If you ever need someone to talk to then please do get in touch. You can find much more information on our website too. Thank you for speaking with me.'],
        'context_filter': ['infoReq', 'advisor'],
        'context_set': ['advisor']
    }, {
        'tag': 'detailsRetrieve',
        'patterns': [],
        'responses': ['Can I take your name?', 'Can I take your email?', 'Where did this occur?', 'Do you remember what time this happened?', 'Is there a phone number that we could speak to you on?', 'What time may we contact you?'],
        'context_filter': ['name', 'email', 'where', 'time', 'phone', 'contactTime'],
        'context_set': ['retrieveDetails']
    }, {
        'tag': 'learnMore',
        'patterns': [],
        'responses': ['Would like to know about your rights in this situation?', 'I think there are some good videos on our website that you might like. Would you like to see them?'],
        'context_filter': ['learnMore'],
        'context_set': ['infoReq']
    }, {
        'tag': 'contactTime',
        'patterns': [],
        'responses': ['Okay, I have all the information I need. Thank you for talking with me today, an advisor will be in touch with you soon.'],
        'context_filter': ['contactTime']
    }]
}

def loadIntents():
    regex = re.compile(r'(?:\[)(.*)(?:\])\s?(.*)\s?', re.IGNORECASE)
    files = [f for f in os.listdir(ROOT_PATH) if f.endswith('.txt')]
    for file in files:
        with open(os.path.join(ROOT_PATH, file), 'r') as open_file:
            for line in open_file:
                if file == 'statements.txt':
                    matches = re.search(regex, line)
                    statement = matches.group(2).replace('{answer}', '')
                    
                    json_output['intents'][0]['patterns'].append(statement)
                elif file == 'greetings.txt':
                    json_output['intents'][1]['patterns'].append(line)
                elif file == 'yes.txt':
                    json_output['intents'][2]['patterns'].append(line)
                elif file == 'no.txt':
                    json_output['intents'][3]['patterns'].append(line)

if __name__ == '__main__':
    loadIntents()
    with open(os.path.join(ROOT_PATH, 'output.json'), 'w') as open_file:
        json.dump(json_output, open_file)