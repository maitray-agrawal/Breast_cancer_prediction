"""
Cancer Health Prediction System - Prediction Script
===================================================
This script loads the trained model and allows users to make predictions
on whether a tumor is benign or malignant based on input features.

Author: AI/ML Project
Date: 2026
"""

# Import required libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')


def load_model_and_scaler():
    """
    Load the trained model and scaler from pickle files.
    
    Returns:
        tuple: (model, scaler, feature_names)
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        # Load the model
        with open('best_cancer_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Load feature names from the dataset
        cancer_data = load_breast_cancer()
        feature_names = cancer_data.feature_names
        target_names = cancer_data.target_names
        
        # Create and fit scaler (for consistency with training)
        X_train = cancer_data.data
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        return model, scaler, feature_names, target_names
    
    except FileNotFoundError:
        print("✗ Error: Model file 'best_cancer_model.pkl' not found!")
        print("Please run 'python train_model.py' first to train and save the model.")
        raise
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise


def display_feature_guide(feature_names):
    """
    Display a guide for understanding the features.
    
    Args:
        feature_names (array): Names of all features
    """
    print("\n" + "=" * 80)
    print("FEATURE GUIDE")
    print("=" * 80)
    print("\nPlease enter values for the following 30 features:")
    print("(Note: All features should be numerical values)")
    print("-" * 80)
    
    for idx, feature in enumerate(feature_names, 1):
        print(f"{idx:2d}. {feature}")
    
    print("-" * 80)


def get_user_input(feature_names):
    """
    Accept user input for all required features with validation.
    
    Args:
        feature_names (array): Names of all features
    
    Returns:
        array: Array of feature values
    
    Raises:
        ValueError: If input is invalid
    """
    print("\n" + "=" * 80)
    print("ENTER FEATURE VALUES")
    print("=" * 80)
    
    features = []
    
    for idx, feature_name in enumerate(feature_names, 1):
        while True:
            try:
                # Get user input
                value = input(f"\n{idx}. {feature_name}: ")
                
                # Validate input
                if value.strip() == '':
                    print("   ✗ Error: Please enter a valid number")
                    continue
                
                # Convert to float
                feature_value = float(value)
                features.append(feature_value)
                print(f"   ✓ Accepted: {feature_value}")
                break
            
            except ValueError:
                print(f"   ✗ Error: '{value}' is not a valid number. Please enter a numerical value.")
            except KeyboardInterrupt:
                print("\n\n✗ Input cancelled by user.")
                raise
    
    return np.array(features).reshape(1, -1)


def make_prediction(model, scaler, features, target_names):
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained machine learning model
        scaler: StandardScaler for feature normalization
        features (array): Input features (1D array)
        target_names (array): Names of target classes
    
    Returns:
        tuple: (prediction_class, confidence_score)
    """
    try:
        # Scale the features using the scaler
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability/confidence
        if hasattr(model, 'predict_proba'):
            # If model has probability prediction (most classifiers do)
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
        else:
            # For models like SVM that don't have predict_proba
            confidence = 0.5
        
        return prediction, confidence
    
    except Exception as e:
        print(f"\n✗ Error during prediction: {str(e)}")
        raise


def display_results(prediction, confidence, target_names):
    """
    Display prediction results in a formatted manner.
    
    Args:
        prediction (int): Predicted class (0 or 1)
        confidence (float): Confidence score (0-1)
        target_names (array): Names of target classes
    """
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    # Get prediction class name
    class_name = target_names[prediction]
    confidence_percentage = confidence * 100
    
    # Display results
    print(f"\n📊 Prediction: {class_name.upper()}")
    print(f"   Confidence Score: {confidence:.4f} ({confidence_percentage:.2f}%)")
    
    # Color-coded display based on prediction
    if class_name == 'benign':
        print("\n   ✓ The tumor is likely BENIGN (non-cancerous)")
        print("   This is generally a positive outcome.")
    else:
        print("\n   ⚠ The tumor is likely MALIGNANT (cancerous)")
        print("   Please consult a medical professional for further evaluation.")
    
    print("\n" + "=" * 80)


def run_prediction_loop(model, scaler, feature_names, target_names):
    """
    Run the prediction system in a loop, allowing multiple predictions.
    
    Args:
        model: Trained model
        scaler: StandardScaler instance
        feature_names (array): Names of features
        target_names (array): Names of target classes
    """
    while True:
        try:
            # Option 1: Manual input
            print("\n\nOptions:")
            print("1. Enter feature values manually")
            print("2. Exit")
            
            choice = input("\nSelect option (1 or 2): ").strip()
            
            if choice == '2':
                print("\n✓ Exiting prediction system. Thank you!")
                break
            elif choice == '1':
                # Get user input
                features = get_user_input(feature_names)
                
                # Make prediction
                prediction, confidence = make_prediction(model, scaler, features, target_names)
                
                # Display results
                display_results(prediction, confidence, target_names)
            else:
                print("✗ Invalid choice. Please select 1 or 2.")
        
        except KeyboardInterrupt:
            print("\n\n✓ Prediction system terminated by user.")
            break
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            print("Please try again.")


def main():
    """
    Main function to run the prediction system.
    """
    try:
        print("\n" + "=" * 80)
        print("CANCER HEALTH PREDICTION SYSTEM - COMMAND LINE INTERFACE")
        print("=" * 80)
        
        # Load model and scaler
        print("\nLoading trained model...")
        model, scaler, feature_names, target_names = load_model_and_scaler()
        print("✓ Model loaded successfully!")
        
        # Display feature guide
        display_feature_guide(feature_names)
        
        # Run prediction loop
        run_prediction_loop(model, scaler, feature_names, target_names)
    
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        print("Please ensure the model has been trained by running 'python train_model.py'")


if __name__ == "__main__":
    main()
