import click
from flask.cli import with_appcontext
from .db import get_db, close_db
from werkzeug.security import generate_password_hash
@click.command("add-user")
@click.option('--adm', prompt="admin password")
@click.option('--un',prompt="username")
@click.option('--pwd', prompt="password")
@with_appcontext
def add_user(admin, username,password):
    # add a user ....
    if admin != 'test':
        return
    db = get_db()
    msg = None
    if not username:
        msg = 'username is required'
    elif not password:
        msg = 'password is required'
    elif db.execute('SELECT id FROM user WHERE username = ?', (username,)).fetchone() is not None:
        msg = 'user {} is already registered'.format(username)
    if msg is None:
        db.execute(
            'INSERT INTO user (username, password) VALUES (?, ?)',
            (username, generate_password_hash(password))
        )
        msg = "successfully added a user"
        db.commit()
        close_db()
    print(msg)
@click.command("delete-user")
@click.option('--adm', prompt='admin password')
@click.option('--un',prompt="username")
@with_appcontext
def delete_user(admin,username):
    if admin != 'test':
        return
    db = get_db()
    msg = None
    if not username:
        msg = 'username is required'
    elif db.execute('SELECT id FROM user WHERE username = ?', (username,)).fetchone() is not None:
        msg = 'user {} doesn\'t exist'.format(username)

    if msg is None:
        db.execute(
            'DELETE  FROM user WHERE username = ?',
            (username,)
        )
        db.commit()
        close_db()
        msg = "successfully deleted"
    print(msg)

def init_app(app):
    app.cli.add_command(add_user)
    app.cli.add_command(delete_user)