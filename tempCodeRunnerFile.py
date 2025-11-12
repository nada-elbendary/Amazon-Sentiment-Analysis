from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

# تحميل الموديل والـ vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    print("✓ تم تحميل الموديل والـ vectorizer بنجاح")
except Exception as e:
    print(f"✗ خطأ في تحميل الملفات: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'الموديل غير متاح حالياً'
            }), 500
        
        data = request.get_json()
        review_text = data.get('review', '')
        
        if not review_text.strip():
            return jsonify({
                'error': 'الرجاء إدخال نص المراجعة'
            }), 400
        
        # تحويل النص إلى features
        text_vectorized = vectorizer.transform([review_text])
        
        # التنبؤ بالنتيجة
        prediction = model.predict(text_vectorized)[0]
        
        # الحصول على احتمالية التنبؤ
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probabilities) * 100)
            
            # إذا كان الموديل binary classification
            if len(probabilities) == 2:
                positive_prob = float(probabilities[1] * 100)
                negative_prob = float(probabilities[0] * 100)
            else:
                positive_prob = confidence if prediction == 1 else 100 - confidence
                negative_prob = 100 - positive_prob
        else:
            # إذا لم يدعم الموديل predict_proba
            confidence = 85.0
            positive_prob = 85.0 if prediction == 1 else 15.0
            negative_prob = 100 - positive_prob
        
        # تحديد المشاعر
        sentiment = 'positive' if prediction == 1 else 'negative'
        sentiment_ar = 'إيجابية' if prediction == 1 else 'سلبية'
        
        return jsonify({
            'sentiment': sentiment,
            'sentiment_ar': sentiment_ar,
            'confidence': round(confidence, 2),
            'positive_percentage': round(positive_prob, 2),
            'negative_percentage': round(negative_prob, 2),
            'review_length': len(review_text.split())
        })
    
    except Exception as e:
        return jsonify({
            'error': f'حدث خطأ في التحليل: {str(e)}'
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)