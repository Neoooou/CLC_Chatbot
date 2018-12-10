from flask import Flask
from flask import request
from chatbot.chatbot_model import ChatBot
app = Flask(__name__)



@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
