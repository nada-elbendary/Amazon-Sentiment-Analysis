<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,18,20,24&height=200&section=header&text=Amazon%20Sentiment%20Analysis&fontSize=50&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Review%20Classification%20System&descAlignY=60&descSize=20" width="100%"/>
</div>

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
  ![NLTK](https://img.shields.io/badge/NLTK-85C1E9?style=for-the-badge)
  
  [![Accuracy](https://img.shields.io/badge/Accuracy-91.13%25-success?style=for-the-badge&logo=chartdotjs)](.)
  [![Dataset](https://img.shields.io/badge/Dataset-500K+%20Reviews-blue?style=for-the-badge&logo=kaggle)](.)
  [![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](.)

  <h3>🧠 Advanced Machine Learning System for Sentiment Analysis</h3>
  <p>Trained on 500,000+ Amazon reviews with 91.13% accuracy using state-of-the-art NLP techniques</p>

</div>

---

## 🎯 Project Overview

This project implements a **high-performance sentiment analysis system** that classifies Amazon product reviews as **Positive** or **Negative**. Built with advanced Natural Language Processing (NLP) and Machine Learning techniques, it achieves **91.13% accuracy** on real-world data.

### ✨ Key Highlights

- 🎯 **91.13% Classification Accuracy**
- 📊 **Trained on 500,000+ Reviews**
- 🧠 **Logistic Regression with TF-IDF**
- 🚀 **Real-time Flask API**
- 💫 **Beautiful Interactive UI**
- ⚡ **Optimized Performance**

---

## 🎥 Demo

<div align="center">
  
  ### 🌐 **[Try Live Demo →](your-deployed-link-here)**
  
  <img src="https://via.placeholder.com/800x400/1a1a1a/8a2be2?text=Add+Your+Screenshot+Here" alt="Demo Screenshot" width="80%"/>
  
</div>

---

## 🏗️ Architecture

```mermaid
graph LR
    A[User Input] --> B[Text Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[Logistic Regression Model]
    D --> E[Sentiment Prediction]
    E --> F[Result Display]
    
    style A fill:#8a2be2,stroke:#fff,stroke-width:2px,color:#fff
    style E fill:#00ff87,stroke:#fff,stroke-width:2px,color:#000
```

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core Language | 3.8+ |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ML Framework | 1.3.0 |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) | Web Framework | 2.3.0 |
| ![NLTK](https://img.shields.io/badge/NLTK-85C1E9?style=flat-square) | NLP Processing | 3.8.0 |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data Processing | 2.0.0 |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical Computing | 1.24.0 |

### Machine Learning Pipeline

```python
Text Input → Preprocessing → TF-IDF → Logistic Regression → Prediction
```

---

## 📊 Model Performance

<div align="center">

### 🎯 Classification Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **91.13%** |
| **Precision** | 91% |
| **Recall** | 91% |
| **F1-Score** | 91% |

</div>

### 📈 Model Comparison

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| **Logistic Regression (C=2.0)** ⭐ | **91.13%** | 22.13s |
| Logistic Regression (C=5.0) | 91.07% | 23.42s |
| Linear SVC | 90.76% | 20.40s |
| Naive Bayes | 87.91% | 0.17s |

### 🔬 Detailed Classification Report

```
              precision    recall  f1-score   support

   Negative       0.91      0.91      0.91     25155
   Positive       0.91      0.91      0.91     24845

   accuracy                           0.91     50000
  macro avg       0.91      0.91      0.91     50000
weighted avg       0.91      0.91      0.91     50000
```

---

## ⚡ Features

<table>
<tr>
<td width="50%">

### 🎨 User Interface
- ✨ Modern glassmorphism design
- 🌈 Animated gradient backgrounds
- 📱 Fully responsive layout
- 🎭 Real-time sentiment display
- 💫 Smooth transitions & effects

</td>
<td width="50%">

### 🧠 ML Capabilities
- 🎯 91.13% accuracy rate
- ⚡ Real-time predictions
- 📊 TF-IDF with 50K features
- 🔄 Advanced text preprocessing
- 🎲 Ensemble learning support

</td>
</tr>
<tr>
<td width="50%">

### 🚀 API Features
- 🔌 RESTful API endpoints
- 📝 JSON request/response
- 🧪 Built-in test suite
- 📊 Confidence scores
- 🛡️ Error handling

</td>
<td width="50%">

### 🔧 Technical Features
- 🧹 Advanced text cleaning
- 📚 NLTK preprocessing
- 🔤 Lemmatization
- 🚫 Stop words removal
- 📈 N-gram analysis

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1️⃣ **Clone the repository**

```bash
git clone https://github.com/nada-elbendary/Amazon-Sentiment-Analysis.git
cd Amazon-Sentiment-Analysis
```

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Download NLTK data**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

4️⃣ **Run the application**

```bash
python app.py
```

5️⃣ **Open your browser**

```
http://localhost:5000
```

---

## 📁 Project Structure

```
Amazon-Sentiment-Analysis/
│
├── 📄 app.py                          # Flask application
├── 🎨 templates/
│   └── index.html                     # Frontend UI
├── 🧠 final_sentiment_model.pkl       # Trained ML model
├── 📊 final_tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── 📓 amazon_review_classifier_py.ipynb  # Training notebook
├── 📋 requirements.txt                # Dependencies
└── 📖 README.md                       # Documentation
```

---

## 💻 Usage

### Web Interface

1. Open `http://localhost:5000` in your browser
2. Enter a product review in the text area
3. Click **"Analyze Sentiment"**
4. View the prediction with confidence score

### API Usage

#### Predict Sentiment

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing!"}'
```

**Response:**

```json
{
  "original_text": "This product is amazing!",
  "cleaned_text": "product amazing",
  "prediction_label": 2,
  "sentiment": "Positive",
  "emoji": "😊",
  "confidence": 95.23,
  "status": "success"
}
```

#### Run Test Suite

```bash
curl http://localhost:5000/api/test
```

#### API Information

```bash
curl http://localhost:5000/api/info
```

---

## 🔬 How It Works

### 1️⃣ **Text Preprocessing**

```python
def clean_text_advanced(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(punctuation_table)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords & apply lemmatization
    words = [lemmatizer.lemmatize(word) 
             for word in text.split() 
             if word not in stop_words]
    
    return " ".join(words)
```

### 2️⃣ **Feature Extraction**

- **TF-IDF Vectorization** with 50,000 features
- **N-grams**: Unigrams + Bigrams (1-2 words)
- **Min/Max Document Frequency**: 3 / 0.9
- **Sublinear TF scaling** for better performance

### 3️⃣ **Model Training**

- **Algorithm**: Logistic Regression (C=2.0)
- **Solver**: SAGA optimizer
- **Training Data**: 500,000 reviews
- **Test Data**: 50,000 reviews
- **Classes**: Binary (Negative=1, Positive=2)

### 4️⃣ **Prediction Pipeline**

```
Input Text → Clean → Vectorize → Model → Confidence Score → Result
```

---

## 📊 Dataset

### 📦 Source
- **Name**: Amazon Review Polarity Dataset
- **Source**: Kaggle
- **Total Size**: 4,000,000 reviews
- **Training Set**: 500,000 reviews (sampled)
- **Test Set**: 50,000 reviews (sampled)

### 📈 Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Positive (2) | ~250,000 | 50% |
| Negative (1) | ~250,000 | 50% |

### 🔍 Sample Data

```
Positive Review: "This product is absolutely amazing! Best purchase ever!"
Negative Review: "Terrible quality. Waste of money. Very disappointed."
```

---

## 🎨 UI Preview

<div align="center">

### Main Interface

<img src="https://via.placeholder.com/700x400/0a0a0a/8a2be2?text=🧠+Neural+Sentiment+Engine" alt="Main Interface"/>

### Analysis Result

<img src="https://via.placeholder.com/700x400/0a0a0a/00ff87?text=✅+Positive+Sentiment+Detected" alt="Result Display"/>

### Test Suite

<img src="https://via.placeholder.com/700x400/0a0a0a/1e90ff?text=🧪+Test+Results+Dashboard" alt="Test Suite"/>

</div>

---

## 🔧 Configuration

### Model Parameters

```python
# TF-IDF Configuration
max_features = 50000
ngram_range = (1, 2)
min_df = 3
max_df = 0.9

# Logistic Regression
C = 2.0
solver = 'saga'
max_iter = 1000
```

### Flask Configuration

```python
host = '0.0.0.0'
port = 5000
debug = True
```

---

## 🧪 Testing

### Run All Tests

```bash
# Via Web Interface
Click "⚡ Test Suite" button

# Via API
curl http://localhost:5000/api/test
```

### Test Cases

```python
test_reviews = [
    "This product is absolutely amazing! I love it!",  # → Positive
    "Terrible quality, waste of money.",                # → Negative
    "It's okay, nothing special.",                     # → Negative
    "Best purchase ever! Highly recommended!",         # → Positive
    "Disappointed and frustrated."                     # → Negative
]
```

---

## 📈 Performance Optimization

### Current Optimizations

✅ **Sparse Matrix Operations** - Efficient memory usage  
✅ **Parallel Processing** - Multi-core training  
✅ **Sublinear TF Scaling** - Better feature normalization  
✅ **Optimized Vectorization** - Fast inference  
✅ **Caching** - Quick repeated predictions  

### Speed Benchmarks

| Operation | Time |
|-----------|------|
| Single Prediction | ~5ms |
| Batch (100 reviews) | ~50ms |
| Model Loading | ~200ms |

---

## 🛣️ Roadmap

### ✅ Completed
- [x] Basic sentiment analysis
- [x] Flask API implementation
- [x] Modern UI with animations
- [x] Model optimization (91.13% accuracy)
- [x] API documentation

### 🚧 In Progress
- [ ] Deploy to production (Render/Heroku)
- [ ] Add model explainability (LIME/SHAP)
- [ ] Multi-language support

### 🎯 Future Plans
- [ ] Real-time streaming analysis
- [ ] Deep Learning models (BERT/Transformers)
- [ ] Aspect-based sentiment analysis
- [ ] Mobile app (React Native)
- [ ] Chrome extension

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Requirements

```txt
Flask==2.3.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
nltk==3.8.0
gunicorn==21.2.0
```

---

## 🐛 Known Issues

- Large model file size (~2MB) - considering model compression
- NLTK data download required on first run
- Limited to English language reviews

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

### **Nada Mohammed Elbendary**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nada-mohammed5)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nadaelbendary3@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nada-elbendary)

</div>

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Amazon Review Polarity](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
- Inspiration: Real-world e-commerce applications
- Tools: Scikit-learn, NLTK, Flask

---

## 📊 Project Stats

<div align="center">

![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-2000+-blue?style=for-the-badge)
![Files](https://img.shields.io/badge/Files-15+-green?style=for-the-badge)
![Size](https://img.shields.io/badge/Size-5MB-orange?style=for-the-badge)

</div>

---

## ⭐ Support

If you found this project helpful, please consider:

- ⭐ **Starring** the repository
- 🐛 **Reporting bugs** or suggesting features
- 🔀 **Forking** and contributing
- 📢 **Sharing** with others

---

<div align="center">
  
### 💡 **Made with ❤️ and Machine Learning**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,18,20,24&height=100&section=footer" width="100%"/>

</div>
