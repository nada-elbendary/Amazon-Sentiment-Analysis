# 🧠 Amazon Sentiment Analysis - AI Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-91.13%25-success.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.0-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Sentiment%20Analysis-purple?style=for-the-badge" alt="ML Badge"/>
  <img src="https://img.shields.io/badge/NLP-NLTK-orange?style=for-the-badge" alt="NLP Badge"/>
</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Screenshots](#-screenshots)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Overview

An advanced **Machine Learning sentiment analysis system** designed to classify Amazon product reviews into positive or negative sentiments. The project achieves **91.13% accuracy** using ensemble machine learning techniques and provides a modern web interface for real-time predictions.

### Problem Statement
Understanding customer sentiment from product reviews is crucial for:
- **E-commerce platforms** to improve product recommendations
- **Businesses** to gauge customer satisfaction
- **Marketers** to analyze brand perception

### Solution
Built a robust ML pipeline that:
- Processes raw text using advanced NLP techniques
- Trains multiple classifiers and selects the best performer
- Deploys via Flask API with an intuitive UI
- Provides real-time sentiment predictions with confidence scores

## ✨ Key Features

- ✅ **High Accuracy**: 91.13% classification accuracy on test data
- ✅ **Large Dataset**: Trained on 500,000 Amazon reviews
- ✅ **Multiple Models**: Logistic Regression, Linear SVC, Naive Bayes, Ensemble
- ✅ **Advanced NLP**: Text preprocessing with NLTK (lemmatization, stopword removal)
- ✅ **TF-IDF Vectorization**: 50,000 features with bigram support
- ✅ **REST API**: Flask-based API for easy integration
- ✅ **Modern UI**: Futuristic, animated web interface
- ✅ **Real-time Predictions**: Instant sentiment analysis with confidence scores
- ✅ **Production Ready**: Optimized for deployment

## 🛠️ Tech Stack

### Machine Learning & Data Processing
- **Python 3.8+**
- **Scikit-learn** - ML models and evaluation
- **NLTK** - Natural Language Processing
- **Pandas & NumPy** - Data manipulation
- **SciPy** - Sparse matrix operations

### Web Framework
- **Flask** - Backend API
- **HTML5/CSS3** - Frontend
- **JavaScript** - Interactive UI

### Development Tools
- **Google Colab** - Model training
- **Kaggle** - Dataset source
- **Pickle** - Model serialization

## 📊 Model Performance

### Comparison of Algorithms

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Logistic Regression (C=2.0)** | **91.13%** | 0.91 | 0.91 | 0.91 | 22.13s |
| Logistic Regression (C=5.0) | 91.07% | 0.91 | 0.91 | 0.91 | 23.42s |
| Linear SVC (C=1.0) | 90.76% | 0.91 | 0.91 | 0.91 | 20.40s |
| Naive Bayes (alpha=0.1) | 87.91% | 0.88 | 0.88 | 0.88 | 0.17s |

### Dataset Statistics
- **Training Set**: 500,000 reviews
- **Test Set**: 50,000 reviews
- **Features**: 50,000 TF-IDF features
- **Classes**: Binary (Positive/Negative)
- **Balance**: ~50/50 split

### Classification Report (Best Model)
```
              precision    recall  f1-score   support

           1       0.91      0.91      0.91     25155
           2       0.91      0.91      0.91     24845

    accuracy                           0.91     50000
   macro avg       0.91      0.91      0.91     50000
weighted avg       0.91      0.91      0.91     50000
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/nada-elbendary/Amazon-Sentiment-Analysis.git
cd amazon-sentiment-analysis
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 5: Verify Installation
```bash
python -c "import flask, sklearn, nltk, pandas; print('All packages installed successfully!')"
```

## 💻 Usage

### Running the Web Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Access the application:**
   - Open browser and go to: `http://127.0.0.1:5000`

3. **Using the interface:**
   - Enter your review text in the textarea
   - Click "Analyze Sentiment" button
   - View results with confidence scores
   - Try the "Test Suite" button for sample predictions

### Using the API

#### Predict Sentiment
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing! I love it!"}'
```

**Response:**
```json
{
  "original_text": "This product is amazing! I love it!",
  "cleaned_text": "product amazing love",
  "prediction_label": 2,
  "sentiment": "Positive",
  "emoji": "😊",
  "confidence": 95.67,
  "status": "success"
}
```

#### Run Tests
```bash
curl http://127.0.0.1:5000/api/test
```

#### Get API Info
```bash
curl http://127.0.0.1:5000/api/info
```

### Training Your Own Model

If you want to retrain the model with different parameters:

1. Open `amazon_review_classifier_py.ipynb` in Google Colab
2. Upload your Kaggle API key
3. Run all cells sequentially
4. Download the generated `.pkl` files
5. Replace `final_sentiment_model.pkl` and `final_tfidf_vectorizer.pkl`




## 📁 Project Structure

CONTENT_REVIEW_ML_PROJECT/
│
├── app.py                          # Flask application
├── final_sentiment_model.pkl       # Trained ML model
├── final_tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── templates/
    └── index.html                  # Web interface
```

```

## 📡 API Documentation

### Endpoints

#### `GET /`
Returns the main web interface

#### `POST /api/predict`
Analyze sentiment of a review

**Request Body:**
```json
{
  "review": "Your review text here"
}
```

**Response:**
```json
{
  "original_text": "string",
  "cleaned_text": "string",
  "prediction_label": 1 or 2,
  "sentiment": "Positive" or "Negative",
  "emoji": "😊" or "😞",
  "confidence": float,
  "status": "success"
}
```

#### `GET /api/test`
Run predefined test cases

**Response:**
```json
{
  "test_results": [...],
  "total_tests": int
}
```

#### `GET /api/info`
Get API information and available endpoints

## 📸 Screenshots

### Main Interface
![Main Interface](screenshots/main-interface.png)
*Modern, animated UI with neural-inspired design*

### Positive Sentiment Analysis
![Positive Result](screenshots/positive-result.png)
*Example of positive review classification*

### Negative Sentiment Analysis
![Negative Result](screenshots/negative-result.png)
*Example of negative review classification*

### Test Suite Results
![Test Suite](screenshots/test-suite.png)
*Batch testing with multiple samples*

## 🎓 What I Learned

### Technical Skills
- **NLP Preprocessing**: Text cleaning, tokenization, lemmatization, stopword removal
- **Feature Engineering**: TF-IDF vectorization with bigrams and n-grams
- **Model Selection**: Comparing multiple algorithms and hyperparameter tuning
- **Ensemble Methods**: Voting classifiers for improved accuracy
- **API Development**: Building RESTful APIs with Flask
- **Frontend Development**: Creating modern, animated interfaces

### Best Practices
- **Data Sampling**: Efficient handling of large datasets
- **Model Serialization**: Using pickle for model deployment
- **Error Handling**: Robust exception handling in production
- **Code Organization**: Modular, maintainable code structure
- **Documentation**: Writing clear, comprehensive documentation

### Challenges Overcome
- Handling large dataset (3.6M+ reviews) with limited resources
- Optimizing text preprocessing for speed and accuracy
- Balancing model complexity vs performance
- Creating responsive UI with smooth animations

## 🔮 Future Improvements

### Short-term
- [ ] Add confidence score visualization (progress bars)
- [ ] Implement more test cases
- [ ] Add export functionality (PDF/CSV reports)
- [ ] Create Docker container for easy deployment

### Medium-term
- [ ] Deploy to cloud platform (AWS/Heroku/Railway)
- [ ] Add multilingual support (Arabic, Spanish, French)
- [ ] Implement user authentication and history
- [ ] Add batch processing for multiple reviews
- [ ] Create data visualization dashboard

### Long-term
- [ ] Integrate deep learning models (BERT, RoBERTa)
- [ ] Add aspect-based sentiment analysis
- [ ] Develop mobile application (React Native)
- [ ] Implement real-time monitoring and logging
- [ ] Add A/B testing framework for model comparison

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Nada Mohammed Ahmed**
- LinkedIn: [Nada Mohammed](https://www.linkedin.com/in/nada-mohammed5)
- Email: nadaelbendary3@gmail.com
- GitHub: [@nada-elbendary](https://github.com/nada-elbendary)


## 🙏 Acknowledgments

- **Dataset**: [Amazon Review Polarity Dataset on Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
- **Inspiration**: Research papers on sentiment analysis and NLP
- **Tools**: Google Colab, Scikit-learn, NLTK, Flask communities

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=nada-elbendary/Amazon-Sentiment-Analysis&type=Date)](https://star-history.com/#nada-elbendary/Amazon-Sentiment-Analysis&Date)

---

<div align="center">
  <p>Made with ❤️ and ☕</p>
  <p>© 2025 Nada Mohammed Elbendary. All rights reserved.</p>
</div>