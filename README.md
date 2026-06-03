# Breast Cancer Prediction System

## Overview

A full-stack machine learning application that predicts malignancy of breast tumors using clinical diagnostic features. The system implements an end-to-end ML pipeline with model deployment, REST APIs, and an interactive web interface for real-time predictions and batch processing.

## Problem Statement

Early and accurate diagnosis of breast cancer is critical for treatment outcomes. This project addresses the need for automated, data-driven decision support systems by leveraging diagnostic imaging metrics to classify tumors with high accuracy and reliability.

## Key Features

- **Multiple Classification Models**: Logistic Regression, Random Forest, Support Vector Machine, and K-Nearest Neighbors with comparative performance analysis
- **Production-Ready Pipeline**: Complete ML lifecycle from data preprocessing through model persistence and API deployment
- **Advanced Model Optimization**: Feature selection, cross-validation, hyperparameter tuning, and class imbalance handling
- **REST API**: Scalable backend architecture enabling programmatic predictions and batch operations
- **Interactive Dashboard**: Real-time web interface for single predictions with visualization and model insights
- **Batch Processing**: CSV-based predictions for multiple samples
- **Model Tracking**: Comprehensive performance metrics and prediction history logging
- **Modular Architecture**: Separation of concerns with reusable components and clear interfaces

## Technical Stack

**Language & Core Libraries**
- Python 3.x
- Scikit-Learn (model training and evaluation)
- NumPy, Pandas (numerical computing and data manipulation)
- Imbalanced-Learn (handling class imbalance)

**Backend & API**
- Flask (web framework)
- SQLite (data persistence)

**Data Visualization**
- Matplotlib, Seaborn

**Development & Version Control**
- Git, GitHub
- Virtual Environments (dependency isolation)

## Project Structure

```
Cancer_Prediction_Project/
├── app.py                           # Flask application entry point
├── app_advanced.py                  # Advanced dashboard features
├── model_metadata.json              # Model configuration and metadata
├── requirements.txt                 # Python dependencies
├── sample_health_dataset.csv        # Example dataset for testing
├── train_model.py                   # Standard model training pipeline
├── train_advanced.py                # Advanced training with optimization
├── train_custom_dataset.py          # Custom dataset training module
├── predict.py                       # Prediction interface
├── test_model.py                    # Model evaluation and testing
├── templates/                       # HTML templates
│   ├── index.html                  # Dashboard interface
│   └── advanced_dashboard.html      # Advanced analytics view
└── static/                          # Static assets
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/maitray-agrawal/Breast_cancer_prediction.git
cd Cancer_Prediction_Project
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

**Standard model training:**
```bash
python train_model.py
```

**Advanced training with optimization:**
```bash
python train_advanced.py
```

**Custom dataset training:**
```bash
python train_custom_dataset.py
```

### Running the Web Application

```bash
python app.py
```
Access the application at `http://localhost:5000`

### Making Predictions

**Single prediction via Python:**
```python
from predict import predict_tumor
result = predict_tumor(features_array)
```

**Batch predictions from CSV:**
See `CUSTOM_DATASET_GUIDE.md` for detailed instructions on preparing and processing batch predictions.

### Model Testing

```bash
python test_model.py
```

## Model Performance

The trained models are evaluated using standard classification metrics including accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation ensures robust performance estimation across data splits.

## Methodology

**Data Preprocessing**
- Feature normalization and scaling
- Missing value handling
- Feature engineering and selection

**Model Development**
- Stratified k-fold cross-validation
- Hyperparameter grid search optimization
- Class imbalance correction using resampling techniques

**Evaluation**
- Comprehensive metrics across training and test datasets
- Performance comparison across model architectures
- Prediction confidence scoring

## Deployment Considerations

- Model serialization for production deployment
- RESTful API for microservice integration
- Containerization-ready architecture
- Scalable design for high-throughput scenarios

## Documentation

For custom dataset preparation and advanced usage, refer to `CUSTOM_DATASET_GUIDE.md`.

## Repository

GitHub: https://github.com/maitray-agrawal/Breast_cancer_prediction

## License

See LICENSE file for details.

## Contributing

Contributions are welcome. Please ensure code follows PEP 8 standards and includes appropriate documentation.

---

**Author**: Maitray Agrawal
