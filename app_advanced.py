"""
Cancer Health Prediction System - ADVANCED Flask Application
============================================================
Enhanced web application with dashboard, batch prediction, 
performance metrics, and prediction history.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import warnings
import os
from io import StringIO

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global variables
model = None
scaler = None
feature_names = None
target_names = None
metadata = None

# Database path
DB_PATH = 'predictions.db'


def init_database():
    """Initialize SQLite database for storing predictions."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_features TEXT,
            prediction TEXT,
            confidence REAL,
            user_ip TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def save_prediction_to_db(features, prediction, confidence, user_ip):
    """Save prediction to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO predictions (timestamp, input_features, prediction, confidence, user_ip)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps([float(f) for f in features]),
            prediction,
            confidence,
            user_ip
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {str(e)}")


def load_model_and_resources():
    """Load trained model and resources."""
    global model, scaler, feature_names, target_names, metadata
    
    try:
        # Load model
        if not os.path.exists('best_cancer_model.pkl'):
            return False
        
        with open('best_cancer_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Load scaler
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
        else:
            # Create default scaler
            cancer_data = load_breast_cancer()
            scaler = StandardScaler()
            scaler.fit(cancer_data.data)
        
        # Load metadata
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
        
        # Load feature names
        cancer_data = load_breast_cancer()
        feature_names = cancer_data.feature_names
        target_names = cancer_data.target_names
        
        # Initialize database
        init_database()
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


@app.route('/')
def home():
    """Home page with prediction form."""
    return render_template('advanced_dashboard.html', 
                          features=feature_names,
                          metadata=metadata)


@app.route('/api/dashboard')
def get_dashboard_data():
    """Get dashboard metrics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total predictions
        c.execute('SELECT COUNT(*) FROM predictions')
        total = c.fetchone()[0]
        
        # Recent predictions
        c.execute('''
            SELECT timestamp, prediction, confidence 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent = [
            {
                'timestamp': row[0],
                'prediction': row[1],
                'confidence': round(row[2], 4)
            }
            for row in c.fetchall()
        ]
        
        # Prediction distribution
        c.execute('''
            SELECT prediction, COUNT(*) 
            FROM predictions 
            GROUP BY prediction
        ''')
        distribution = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
        
        return jsonify({
            'success': True,
            'total_predictions': total,
            'recent_predictions': recent,
            'prediction_distribution': distribution,
            'model_info': metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Make single prediction."""
    try:
        data = request.json
        
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'No features provided'}), 400
        
        features = np.array([float(f) for f in data['features']]).reshape(1, -1)
        
        if features.shape[1] != len(feature_names):
            return jsonify({
                'success': False,
                'error': f'Expected {len(feature_names)} features, got {features.shape[1]}'
            }), 400
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][prediction]
        
        # Save to database
        user_ip = request.remote_addr
        save_prediction_to_db(features[0], target_names[prediction], float(confidence), user_ip)
        
        # Prepare response
        class_name = target_names[prediction]
        risk_level = 'High Risk' if class_name == 'malignant' else 'Low Risk'
        recommendation = ('Immediate medical consultation recommended.' 
                         if class_name == 'malignant' 
                         else 'Continue regular check-ups.')
        
        return jsonify({
            'success': True,
            'prediction': class_name.upper(),
            'confidence': round(float(confidence), 4),
            'confidence_percentage': round(float(confidence) * 100, 2),
            'risk_level': risk_level,
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files allowed'}), 400
        
        # Read CSV
        stream = StringIO(file.stream.read().decode('utf-8'))
        df = pd.read_csv(stream)
        
        # Validate
        if df.shape[1] != len(feature_names):
            return jsonify({
                'success': False,
                'error': f'Expected {len(feature_names)} columns'
            }), 400
        
        # Predict
        features_scaled = scaler.transform(df.values)
        predictions = model.predict(features_scaled)
        confidences = model.predict_proba(features_scaled).max(axis=1)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['prediction'] = [target_names[p] for p in predictions]
        results_df['confidence'] = confidences.round(4)
        
        # Save results
        results_df.to_csv('batch_results.csv', index=False)
        
        return jsonify({
            'success': True,
            'message': f'Predictions made for {len(df)} samples',
            'predictions': results_df.to_dict('records')[:10]  # First 10 rows
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download-results')
def download_results():
    """Download batch prediction results."""
    try:
        return send_file('batch_results.csv', as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """Get prediction history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            SELECT timestamp, prediction, confidence 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        
        history = [
            {
                'timestamp': row[0],
                'prediction': row[1],
                'confidence': round(row[2], 4)
            }
            for row in c.fetchall()
        ]
        
        conn.close()
        
        return jsonify({'success': True, 'history': history})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/features')
def get_features():
    """Get feature information."""
    try:
        return jsonify({
            'success': True,
            'feature_count': len(feature_names),
            'features': list(feature_names),
            'model_info': metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_available': os.path.exists(DB_PATH),
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("=" * 70)
    print("CANCER PREDICTION SYSTEM - ADVANCED FLASK APPLICATION")
    print("=" * 70)
    
    if load_model_and_resources():
        print("\n✓ Model and resources loaded successfully!")
        print("\n📊 Starting Flask server...")
        print("🌐 Open http://localhost:5000 in your browser")
        print("\nAPI Endpoints:")
        print("  - GET  /                    Dashboard")
        print("  - POST /predict              Single prediction")
        print("  - POST /batch-predict        CSV batch prediction")
        print("  - GET  /api/dashboard        Dashboard metrics")
        print("  - GET  /api/history          Prediction history")
        print("  - GET  /api/features         Feature info")
        print("  - GET  /api/health           Health check")
        print("\n⚠️  Press CTRL+C to stop")
        print("=" * 70 + "\n")
        
        app.run(debug=True, host='localhost', port=5000, use_reloader=False)
    else:
        print("\n✗ Failed to load model!")
        print("Run: python train_advanced.py")
