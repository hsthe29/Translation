from transformer import load_translator
from flask import Flask

app = Flask(__name__)


def generate_server_state():
    return {"translator": load_translator("assets/config/configV1.json"),
            "running": True}


from . import views
