from flask import Flask, request, jsonify
import pickle
import re
from nltk import PorterStemmer # Porter Stemmer is an algorithm used to reduce English words to their root form by removing suffixes
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

app = Flask(__name__)


with open('./models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./models/gradient_boosting_classifier.pkl', 'rb') as f:
    gbc_model = pickle.load(f)

def predict(text: str):
    corpus = []
    porter_stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ porter_stemmer.stem(word) if word not in STOPWORDS else word for word in review ]
    review = " ".join(review)
    corpus.append(review)
    input_vec = vectorizer.transform(corpus)
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
