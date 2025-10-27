import pickle

from flask import Flask
from flask import request
from flask import jsonify
from fastapi import FastAPI

input_file = f'pipeline_v1.bin'


with open(input_file, 'rb') as f_in:
   dv, model = pickle.load(f_in)

app = Flask('model')



@app.route('/predict', methods=['POST'])

def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    get_result = y_pred >= 0.5

    result = {
        'get_probability': float(y_pred),
        'result': bool(get_result)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888)


# how to run
# poetry run hypercorn q4:app --bind 127.0.0.1:9696
#  poetry run waitress-serve --listen=127.0.0.1:8888 q4:app