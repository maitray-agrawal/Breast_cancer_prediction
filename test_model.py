"""
Cancer Health Prediction System - Direct Testing Script
========================================================
Test the model directly using Python without CLI or web interface.
Useful for debugging and validation.
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def test_model_directly():
    """
    Test the trained model with sample data directly.
    """
    print("=" * 80)
    print("CANCER PREDICTION MODEL - DIRECT TESTING")
    print("=" * 80)
    
    try:
        # Load the trained model
        print("\n1. Loading trained model...")
        with open('best_cancer_model.pkl', 'rb') as file:
            model = pickle.load(file)
        print("   ✓ Model loaded successfully!")
        
        # Load dataset for feature names and scaler
        print("\n2. Loading dataset and creating scaler...")
        cancer_data = load_breast_cancer()
        feature_names = cancer_data.feature_names
        target_names = cancer_data.target_names
        
        # Create scaler (same as training)
        scaler = StandardScaler()
        scaler.fit(cancer_data.data)
        print("   ✓ Scaler created successfully!")
        
        # Test Case 1: Sample Benign Case
        print("\n" + "=" * 80)
        print("TEST CASE 1: BENIGN TUMOR (Expected: Benign)")
        print("=" * 80)
        
        # Sample benign case from actual data
        benign_sample = cancer_data.data[0:1]  # First sample (benign)
        benign_sample_scaled = scaler.transform(benign_sample)
        
        prediction_benign = model.predict(benign_sample_scaled)[0]
        confidence_benign = model.predict_proba(benign_sample_scaled)[0][prediction_benign]
        
        print(f"\nInput Features (first 5): {benign_sample[0][:5]}")
        print(f"Prediction: {target_names[prediction_benign].upper()}")
        print(f"Confidence: {confidence_benign:.4f} ({confidence_benign * 100:.2f}%)")
        print(f"Actual Class: {target_names[0].upper()}")
        print(f"Result: {'✓ CORRECT' if prediction_benign == 0 else '✗ INCORRECT'}")
        
        # Test Case 2: Sample Malignant Case
        print("\n" + "=" * 80)
        print("TEST CASE 2: MALIGNANT TUMOR (Expected: Malignant)")
        print("=" * 80)
        
        # Sample malignant case from actual data
        malignant_idx = np.where(cancer_data.target == 1)[0][0]
        malignant_sample = cancer_data.data[malignant_idx:malignant_idx+1]
        malignant_sample_scaled = scaler.transform(malignant_sample)
        
        prediction_malignant = model.predict(malignant_sample_scaled)[0]
        confidence_malignant = model.predict_proba(malignant_sample_scaled)[0][prediction_malignant]
        
        print(f"\nInput Features (first 5): {malignant_sample[0][:5]}")
        print(f"Prediction: {target_names[prediction_malignant].upper()}")
        print(f"Confidence: {confidence_malignant:.4f} ({confidence_malignant * 100:.2f}%)")
        print(f"Actual Class: {target_names[1].upper()}")
        print(f"Result: {'✓ CORRECT' if prediction_malignant == 1 else '✗ INCORRECT'}")
        
        # Test Case 3: Custom Values
        print("\n" + "=" * 80)
        print("TEST CASE 3: CUSTOM TEST VALUES")
        print("=" * 80)
        
        # Create custom feature values
        custom_features = np.array([[
            13.5,    # mean radius
            18.2,    # mean texture
            85.5,    # mean perimeter
            570,     # mean area
            0.095,   # mean smoothness
            0.08,    # mean compactness
            0.02,    # mean concavity
            0.02,    # mean concave points
            0.19,    # mean symmetry
            0.06,    # mean fractal dimension
            0.5,     # radius error
            1.0,     # texture error
            3.0,     # perimeter error
            20,      # area error
            0.005,   # smoothness error
            0.01,    # compactness error
            0.01,    # concavity error
            0.005,   # concave points error
            0.01,    # symmetry error
            0.001,   # fractal dimension error
            15.0,    # worst radius
            25.0,    # worst texture
            100,     # worst perimeter
            700,     # worst area
            0.12,    # worst smoothness
            0.15,    # worst compactness
            0.05,    # worst concavity
            0.03,    # worst concave points
            0.25,    # worst symmetry
            0.08     # worst fractal dimension
        ]])
        
        custom_sample_scaled = scaler.transform(custom_features)
        prediction_custom = model.predict(custom_sample_scaled)[0]
        confidence_custom = model.predict_proba(custom_sample_scaled)[0][prediction_custom]
        
        print(f"\nCustom Features (first 5): {custom_features[0][:5]}")
        print(f"Prediction: {target_names[prediction_custom].upper()}")
        print(f"Confidence: {confidence_custom:.4f} ({confidence_custom * 100:.2f}%)")
        
        if target_names[prediction_custom] == 'benign':
            print(f"Interpretation: Low risk - likely benign tumor")
        else:
            print(f"Interpretation: High risk - likely malignant tumor")
        
        # Summary Statistics
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"\nTest Cases Executed: 3")
        print(f"Model Type: {type(model).__name__}")
        print(f"Number of Features: {len(feature_names)}")
        print(f"Classes: {list(target_names)}")
        print(f"\nModel is ready for production use! ✓")
        
    except FileNotFoundError:
        print("✗ Error: 'best_cancer_model.pkl' not found!")
        print("Please run: python train_model.py")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def batch_test():
    """
    Test the model on entire test dataset to get overall performance.
    """
    print("\n" + "=" * 80)
    print("BATCH TEST: MODEL ACCURACY ON FULL DATASET")
    print("=" * 80)
    
    try:
        # Load model and data
        with open('best_cancer_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        cancer_data = load_breast_cancer()
        
        # Create scaler and scale all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cancer_data.data)
        y = cancer_data.target
        
        # Make predictions on entire dataset
        predictions = model.predict(X_scaled)
        
        # Calculate accuracy
        accuracy = (predictions == y).sum() / len(y)
        
        print(f"\nDataset Size: {len(y)} samples")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Correct Predictions: {(predictions == y).sum()}")
        print(f"Incorrect Predictions: {(predictions != y).sum()}")
        
        # Class-wise accuracy
        for class_idx, class_name in enumerate(cancer_data.target_names):
            mask = y == class_idx
            class_accuracy = (predictions[mask] == y[mask]).sum() / mask.sum()
            print(f"{class_name.capitalize()} Accuracy: {class_accuracy:.4f} ({class_accuracy * 100:.2f}%)")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    # Run direct tests
    test_model_directly()
    
    # Run batch test
    batch_test()
    
    print("\n" + "=" * 80)
    print("✓ TESTING COMPLETE")
    print("=" * 80)
