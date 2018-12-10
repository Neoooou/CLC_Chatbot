import json
import ctypes

from django.shortcuts import render
from django.shortcuts import HttpResponse
import requests
import logging
access_token = None
# send request to server to get the access token
def auth(username='test', password='test'):
    url = 'http://localhost:8080/chatbot-server/auth'
    r = requests.post(url, json={"username": username, "password": password})
    json_data = r.json()
    global access_token
    _access_token = json_data.get('access_token')
    if _access_token is not None:
        access_token = _access_token
    else:
        logging.error(json_data.get('msg'))
# Send request that gets response for the user_input
def request_answer(user_input):
    url = 'http://localhost:8080/chatbot-server/converse'
    # set the access_token to authorization header
    if access_token is None:
        auth()
    r = requests.post(url,
                      headers={
                            "Content-Type": "application/json",
                            "Authorization": "Bearer {}".format(access_token)
                      },
                      json={"input_text": user_input})
    json_data = r.json()
    return json_data

def send_user_input(request):
#    user_input = 'I was arrested by the police'
    user_input = request.GET.get('user_input')
    response_data= request_answer(user_input)
   # assert response_data['link'] is not None
    #assert 'Recommended paragraph' in response_data['res']
    return HttpResponse(json.dumps(response_data), content_type='application/json')
def index(request):
    return render(request, 'chat/index.html', {})
