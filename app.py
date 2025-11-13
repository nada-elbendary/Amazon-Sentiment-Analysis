from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()


print("Loading the model...")

try:
    with open('final_sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('final_tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    print("Model loaded successfully!")
except FileNotFoundError as e:
    print(f"خطأ: الملفات غير موجودة! تأكد من وجود:")
    print("   - final_sentiment_model.pkl")
    print("   - final_tfidf_vectorizer.pkl")
    exit(1)


def clean_text_advanced(text):
    
   
    
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "unknown"

        text = text.lower()
        text = text.translate(punctuation_table)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        filtered_words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word not in stop_words and len(word) > 2
        ]

        result = " ".join(filtered_words)
        return result if result else "unknown"

    except Exception as e:
        print(f"Error cleaning text: {e}")

        return "unknown"


def predict_sentiment(review_text):
    
    try:
       
        cleaned = clean_text_advanced(review_text)
        
        
        tfidf_vector = tfidf.transform([cleaned])
        
    
        prediction = model.predict(tfidf_vector)[0]
        
        
        try:
            proba = model.predict_proba(tfidf_vector)[0]
            confidence = float(max(proba)) * 100
        except:
            confidence = None
        
        
        sentiment = "Positive" if prediction == 2 else "Negative"
        emoji = "" if prediction == 2 else ""
        
        return {
            "original_text": review_text,
            "cleaned_text": cleaned,
            "prediction_label": int(prediction),
            "sentiment": sentiment,
            "emoji": emoji,
            "confidence": confidence,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/api/info')
def info():
    
    return jsonify({
        "message": "Amazon Sentiment Analysis API",
        "version": "1.0",
        "endpoints": {
            "/": "GET - Main interface",
            "/api/predict": "POST - Analyze review",
            "/api/test": "GET - Sample tests"

        },
        "note": "Label 1 = Negative, Label 2 = Positive"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    
    try:
        data = request.get_json()
        review = data.get('review', '')
        
        if not review or review.strip() == "":
            return jsonify({
                "error": "Review text must be provided",

                "status": "failed"
            }), 400
        
        result = predict_sentiment(review)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/api/test')
def test():
   
    test_reviews = [
        "This product is absolutely amazing! I love it!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special.",
        "Best purchase ever! Highly recommended!",
        "Disappointed and frustrated. Never buying again.",
        "Good value for money",
        "Worst experience ever"
    ]
    
    results = []
    for review in test_reviews:
        result = predict_sentiment(review)
        results.append(result)
    
    return jsonify({
        "test_results": results,
        "total_tests": len(results)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Amazon Sentiment Analysis - Flask Server")
    print("="*60)
    print("Interface: http://127.0.0.1:5000")
    print("API Info: http://127.0.0.1:5000/api/info")
    print("Test: http://127.0.0.1:5000/api/test")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))