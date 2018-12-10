from flask import(
    Blueprint, request, url_for, jsonify, make_response
)
from werkzeug.exceptions import abort
from .chatbot_model import ChatBot
from flask_jwt_extended import (jwt_required, create_access_token, get_jwt_identity)
from .db import get_db, close_db
from werkzeug.security import check_password_hash
import datetime
bp = Blueprint('chatbot', __name__, url_prefix="/chatbot-server")
model = ChatBot()
@bp.route("/auth", methods=['POST'])
def auth():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    username = request.get_json().get('username')
    password = request.get_json().get('password')
    if not username:
        response = make_response(jsonify({'msg':'Missing username parameter'}))
    elif not password:
        response = make_response(jsonify({'msg': 'Missing password parameter'}))
    else:
        #check username and password
        db = get_db()
        user = db.execute('SELECT * FROM user WHERE username = ?', (username, )).fetchone()
        if user is None:
            response = make_response(jsonify({"msg":"Incorrect username"}))
        elif not check_password_hash(user['password'], password):
            response = make_response(jsonify({"msg":"Incorrect password"}))
        else:
            # pass authentication, return access_token
            access_token = create_access_token(identity=username, expires_delta=datetime.timedelta(days=1))
            response = make_response(jsonify({"access_token":access_token}))
        close_db()
    return response

@bp.route('/converse', methods = ['GET','POST'])
@jwt_required
def converse():
    if not request.is_json:
        return jsonify({"msg": "misssing JSON in request"}), 400
    input_text = request.get_json().get('input_text')
    global model
    response, doc, link = model.request(input_text)
    response_data = {'res':response,
                     'link':link,
                     'doc':doc}
    return jsonify(response_data)


