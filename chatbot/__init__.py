import os
from flask import Flask
from flask_jwt_extended import JWTManager
def create_app(test_config = None):
    # create and configure the app
    # instance_relative_config tells app that configuration file are relative to the instance folder
    app = Flask(__name__, instance_relative_config=True)
    app.config['JWT_SECRET_KEY'] = 'test'
    jwt = JWTManager(app)
    app.config.from_mapping(
        # SECRET_KEY is used by Flask and extensions to keep data safe
        # and should be overridden with a random value when deploying
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'chatbot.sqlite'),
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'hello world'
    from . import db
    db.init_app(app)
    from . import chatbot
    app.register_blueprint(chatbot.bp)
    from . import user_manager
    user_manager.init_app(app)
    return app