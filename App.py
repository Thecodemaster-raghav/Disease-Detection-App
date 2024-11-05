from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('disease_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['age'], data['blood_pressure'], data['cholesterol']]
    prediction = model.predict([features])
    return jsonify({'diagnosis': 'Positive' if prediction[0] == 1 else 'Negative'})

if __name__ == '__main__':
    app.run(debug=True)
