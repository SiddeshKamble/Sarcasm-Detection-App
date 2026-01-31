# Sarcasm Detection App 

A comprehensive machine learning application that detects sarcasm in text using multiple ML and deep learning models. Built with Streamlit for an interactive web interface.

## 🚀 Live Demo
https://sarcasm-detection-app.streamlit.app/

<img width="1298" height="643" alt="Screenshot 2026-01-30 at 10 35 32 PM" src="https://github.com/user-attachments/assets/70b17bfc-1781-4148-8816-4c4f777f6427" />



## Features

- **Multiple ML Models**: Naive Bayes, Logistic Regression, Pretrained RoBERTa, and DistilBERT
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Real-time Predictions**: Test sarcasm detection on custom text inputs
- **Model Comparison**: Compare performance across different algorithms

## Dataset

The app uses a Reddit sarcasm detection dataset (`sarcasm.csv`) containing:
- **comment**: Text comments from Reddit
- **label**: Binary labels (1 = sarcasm, 0 = not sarcasm)
- **subreddit**: Source subreddit
- **score**: Reddit score/upvotes
- **author**: Comment author
- **parent_comment**: Parent comment context

    
<img width="1499" height="865" alt="Screenshot 2026-01-30 at 10 35 50 PM" src="https://github.com/user-attachments/assets/bf16a6a6-931e-4348-b3e3-c1bb1bee1b36" />

<img width="1294" height="637" alt="Screenshot 2026-01-30 at 10 36 40 PM" src="https://github.com/user-attachments/assets/5366a4a4-0169-4874-9d1f-54a3f91cfee3" />

## Models Implemented

### 1. **Naive Bayes**
- Uses CountVectorizer for feature extraction
- Fast training and prediction
- Good baseline performance

### 2. **Logistic Regression**
- TF-IDF vectorization with n-grams (1,2)
- Max features: 5000
- Robust linear classifier

### 3. **Pretrained RoBERTa**
- Model: `jkhan447/sarcasm-detection-RoBerta-base-CR`
- Fine-tuned specifically for sarcasm detection
- State-of-the-art transformer architecture

### 4. **DistilBERT**
- Lightweight BERT variant
- Faster inference while maintaining performance
- Good balance between speed and accuracy

## App Sections

### Home
Welcome page with project overview and features

### Dataset Overview
- Dataset preview
- Class distribution visualization
- Basic statistics

### EDA (Exploratory Data Analysis)
- Sarcasm proportion by comment length
- Top words in sarcastic vs non-sarcastic comments
- Subreddit analysis
- Reddit score distributions
- Word clouds for both classes

### Model Training
- Train all models with one click
- Automatic model serialization
- Performance metrics calculation

### Interactive Prediction
- Real-time sarcasm detection
- Test custom sentences
- Compare predictions across all models

## Installation

### Prerequisites
```bash
Python 3.8+
```

### Setup
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sarcasm-detection-app.git
cd sarcasm-detection-app
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
   - Ensure `sarcasm.csv` is in the root directory
   - Dataset should contain columns: `label`, `comment`, `subreddit`, `score`, `author`, `parent_comment`

## Usage

### Running the App
```bash
streamlit run temp.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface
1. Navigate through sections using the sidebar
2. Explore the dataset and visualizations
3. Train models by clicking "Train All Models"
4. Test predictions in the Interactive Prediction section
   
<img width="936" height="686" alt="Screenshot 2026-01-30 at 10 43 57 PM" src="https://github.com/user-attachments/assets/edaceea8-26f6-431d-a98c-28d019f7dcb8" />

### Example Test Sentences
- `"Thank you for your feedback. It WaS ReAlLy InSIgHtFul!"`
- `"Wow, I really didn't expect you to pass that exam. Good for you!!!"`
- `"Methodology: Crafting the Ultimate Seriousness Detector"`


## 📁 Project Structure

```
sarcasm-detection-app/
├── temp.py                          # Main Streamlit application
├── sarcasm.csv                      # Dataset file
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── naive_bayes_model.pkl           # Saved Naive Bayes model
├── logistic_regression_model.pkl   # Saved Logistic Regression model
├── pretrained_transformer_model.pkl # Saved RoBERTa model
└── distilbert_model.pkl            # Saved DistilBERT model
```

## Performance Metrics

The app calculates and displays:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

## Model Comparison

| Model | Strengths | Use Case |
|-------|-----------|----------|
| Naive Bayes | Fast, simple, good baseline | Quick prototyping |
| Logistic Regression | Interpretable, robust | Production baseline |
| RoBERTa | High accuracy, context-aware | Best performance |
| DistilBERT | Balanced speed/accuracy | Production deployment |

## Deployment

### Local Deployment
```bash
streamlit run temp.py --server.port 8501
```

### Cloud Deployment
The app can be deployed on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: With `Procfile` configuration
- **AWS/GCP**: Using container services
- **Docker**: Containerized deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Reddit for providing the sarcasm dataset
- Hugging Face for transformer models
- Streamlit team for the amazing framework
- Open source ML community

---

> "Sarcasm is the lowest form of wit but the highest form of intelligence." – Oscar Wilde


