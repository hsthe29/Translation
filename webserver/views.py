from . import app
from flask import render_template, request, jsonify

from transformer import Translator
from transformer.model import TransformerConfig, GenerationConfig


def load_translator():
    config = TransformerConfig.load("assets/config/configV1.json")
    return Translator("checkpoint/loss/model_weights.pt", config)


translator = load_translator()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template(
        'index.html',
    )


@app.route('/demo')
def view_form():
    return render_template('demo.html')


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    # Do something with the data
    # print(data)
    # data {'input_language': 'english', 'target_language': 'vietnamese', 'text': 'Hihi'}
    
    input_encoding, target_encoding = translator.preprocess_input(data["text"],
                                                                  data["input_language"],
                                                                  data["target_language"])
    
    input_tokens, results = translator.generate(input_ids=input_encoding.ids,
                                                target_in_ids=target_encoding.ids,
                                                input_mask=input_encoding.mask,
                                                config=GenerationConfig(beam_width=5, n_best=3))
    best_result = results[0]
    
    response_data = {
        "input_tokens": input_tokens,
        "target_tokens": best_result["token"],
        "translation": best_result["translation"],
        "weight": best_result["weight"]
    }
    
    return jsonify(response_data), 200
