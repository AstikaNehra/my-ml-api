print("STARTING FLASK APP")
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

print("ADDING CORS HEADERS")

@app.after_request
def add_cors_headers(response):
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

print("ADDED CORS HEADERS")

print("Loading model pkl")
model = joblib.load('model_quantile_gb.pkl')
print("Loading labeler pkl")
labeler = joblib.load('skill_label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data.get('attempt_count'),
        data.get('fraction_of_hints_used'),
        data.get('time_on_task'),
        data.get('problem_completed'),
        data.get('student_answer_count'),
        data.get('main_skill_enc')
    ]
    prediction = model.predict([features])[0]
    return jsonify({
        'prediction': str(prediction),
        'label': str(prediction)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)