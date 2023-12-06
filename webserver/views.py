from flask import render_template, request, jsonify

from transformer.model import GenerationConfig
from . import app, generate_server_state

server = generate_server_state()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template(
        'index.html',
    )


@app.route('/shutdown', methods=['POST'])
def shutdown():
    data = request.get_json()  # {'shutdown': ,'input_language': , 'target_language': , 'text': }
    
    is_shutdown = data["shutdown"]
    
    if is_shutdown:
        server["running"] = False
    
    response_data = {
        "shutdown": True
    }
    
    return jsonify(response_data), 200


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()  # {'shutdown': ,'input_language': , 'target_language': , 'text': }
    
    if server["running"]:
        translator = server["translator"]
        input_encoding, target_encoding = translator.preprocess_input(data["text"],
                                                                      data["input_language"],
                                                                      data["target_language"])
        
        input_tokens, results = translator.generate(input_encoding,
                                                    target_encoding,
                                                    generation_config=GenerationConfig(beam_width=5, n_best=5))
        best_result = results[0]
        
        response_data = {
            "input_tokens": input_tokens,
            "target_tokens": best_result["token"],
            "translation": best_result["translation"],
            "weight": best_result["weight"]
        }
        
        return jsonify(response_data), 200
    
    return jsonify({}), 404
