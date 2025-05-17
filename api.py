from flask import Flask, request, jsonify

app = Flask(__name__)

import pickle
with open('./models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./models/gradient_boosting_classifier.pkl', 'rb') as f:
    gbc_model = pickle.load(f)

def predict(text: str):
    input_vec = vectorizer.transform([text])
    scaled_input = scaler.transform(input_vec)
    prob = gbc_model.predict_proba(scaled_input)[0]
    prediction = "positive" if prob[1] > prob[0] else "negative"
    return prediction, float(prob.max().round(2))

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction, probability = predict(text)
    return jsonify({'prediction': prediction, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
