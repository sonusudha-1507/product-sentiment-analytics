from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
from src.nlp_pipeline import TextProcessor
from src.engine import train_nlp_infrastructure, extract_topic_insights

app = Flask(__name__)

# Guarantee model availability prior to handling client lifecycle configurations
if not os.path.exists('models/sentiment_classifier.pkl'):
    train_nlp_infrastructure()

classifier = joblib.load('models/sentiment_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
processor = TextProcessor()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_stream():
    try:
        payload = request.json
        raw_text = payload.get('text', '').strip()
        
        if not raw_text:
            return jsonify({'success': False, 'error': 'Null token stream encountered.'})
            
        cleaned_text = processor.clean_text(raw_text)
        vectorized = vectorizer.transform([cleaned_text])
        
        probabilities = classifier.predict_proba(vectorized)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        themes = extract_topic_insights(raw_text)
        
        return jsonify({
            'success': True,
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'confidence': round(confidence * 100, 2),
            'themes': themes,
            'token_count': len(cleaned_text.split())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)