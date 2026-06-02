"""
Cancer Health Prediction System - Flask Web Application
========================================================
This Flask application provides a web interface for the cancer prediction model.
Users can input tumor features and get predictions through a user-friendly website.

Author: AI/ML Project
Date: 2026
"""

# Import required libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import warnings
import os

warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables to store model and scaler
model = None
scaler = None
feature_names = None
target_names = None


def load_model_resources():
    """
    Load the trained model, scaler, and dataset information.
    This function is called once when the application starts.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global model, scaler, feature_names, target_names
    
    try:
        # Load the trained model from pickle file
        if not os.path.exists('best_cancer_model.pkl'):
            print("✗ Model file not found. Please run train_model.py first.")
            return False
        
        with open('best_cancer_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Load feature names and target names from dataset
        cancer_data = load_breast_cancer()
        feature_names = cancer_data.feature_names
        target_names = cancer_data.target_names
        
        # Create and fit scaler for feature normalization
        X_train = cancer_data.data
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        print("✓ Model and resources loaded successfully!")
        return True
    
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False


@app.route('/')
def home():
    """
    Route for the home page - displays the prediction form.
    
    Returns:
        Rendered HTML template with feature input form
    """
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Route for making predictions via API endpoint.
    Accepts JSON data with feature values and returns prediction.
    
    Returns:
        JSON response containing prediction and confidence score
    """
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate that we have features
        if not data or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            }), 400
        
        # Extract feature values
        feature_values = data['features']
        
        # Validate that we have exactly 30 features
        if len(feature_values) != len(feature_names):
            return jsonify({
                'success': False,
                'error': f'Expected {len(feature_names)} features, got {len(feature_values)}'
            }), 400
        
        # Validate that all features are numbers
        try:
            features_array = np.array([float(f) for f in feature_values]).reshape(1, -1)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'All feature values must be numbers'
            }), 400
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get confidence score
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 0.5
        
        # Get class name
        class_name = target_names[prediction]
        
        # Determine risk level and recommendation
        if class_name == 'benign':
            risk_level = 'Low'
            recommendation = 'The tumor is likely benign. Continue regular monitoring.'
        else:
            risk_level = 'High'
            recommendation = 'The tumor is likely malignant. Consult a medical professional for further evaluation.'
        
        # Return prediction results
        return jsonify({
            'success': True,
            'prediction': class_name.upper(),
            'confidence': round(confidence, 4),
            'confidence_percentage': round(confidence * 100, 2),
            'risk_level': risk_level,
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/features', methods=['GET'])
def get_features():
    """
    API endpoint to get the list of features.
    Useful for dynamic form generation on the frontend.
    
    Returns:
        JSON response containing feature names and count
    """
    try:
        return jsonify({
            'success': True,
            'feature_count': len(feature_names),
            'features': list(feature_names)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the application is running.
    
    Returns:
        JSON response indicating application status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_available': len(feature_names) if feature_names else 0
    })


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors - page not found.
    
    Args:
        error: The error object
    
    Returns:
        JSON response with error message
    """
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors - internal server error.
    
    Args:
        error: The error object
    
    Returns:
        JSON response with error message
    """
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("CANCER HEALTH PREDICTION SYSTEM - FLASK WEB APPLICATION")
    print("=" * 70)
    
    # Load model and resources before starting the server
    if load_model_resources():
        print("\n" + "=" * 70)
        print("🚀 Starting Flask server...")
        print("=" * 70)
        print("\n📱 Web Application running on: http://localhost:5000")
        print("   Open your browser and navigate to http://localhost:5000")
        print("\n⚠️  Press CTRL+C to stop the server")
        print("=" * 70 + "\n")
        
        # Run Flask application
        # debug=True: Enables auto-reload on code changes and better error messages
        # use_reloader=False: Prevents loading the model twice
        app.run(debug=True, host='localhost', port=5000, use_reloader=False)
    else:
        print("\n✗ Failed to load model. Please ensure 'best_cancer_model.pkl' exists.")
        print("Run 'python train_model.py' to train and save the model first.")
