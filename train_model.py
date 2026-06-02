"""
Cancer Health Prediction System - Model Training Script
========================================================
This script loads the Breast Cancer Wisconsin dataset, performs exploratory data analysis (EDA),
preprocesses the data, trains multiple machine learning models, and saves the best-performing model.

Author: AI/ML Project
Date: 2026
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data():
    """
    Load the Breast Cancer Wisconsin dataset and convert to Pandas DataFrame.
    
    Returns:
        tuple: (X - features, y - target labels, feature_names, target_names)
    """
    print("=" * 70)
    print("LOADING BREAST CANCER WISCONSIN DATASET")
    print("=" * 70)
    
    # Load the dataset from sklearn
    cancer_data = load_breast_cancer()
    
    # Create a DataFrame from the dataset
    X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y = pd.Series(cancer_data.target, name='target')
    
    print(f"\n✓ Dataset loaded successfully!")
    
    return X, y, cancer_data.feature_names, cancer_data.target_names


def exploratory_data_analysis(X, y, target_names):
    """
    Perform comprehensive exploratory data analysis on the dataset.
    Includes: dataset shape, missing values, statistics, correlations, and visualizations.
    
    Args:
        X (DataFrame): Feature data
        y (Series): Target data
        target_names (array): Names of target classes
    """
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # 1. Display dataset shape
    print(f"\n1. Dataset Shape:")
    print(f"   - Number of samples: {X.shape[0]}")
    print(f"   - Number of features: {X.shape[1]}")
    
    # 2. Display column names
    print(f"\n2. Feature Names ({X.shape[1]} total):")
    for idx, col in enumerate(X.columns, 1):
        print(f"   {idx}. {col}")
    
    # 3. Check missing values
    print(f"\n3. Missing Values:")
    missing_count = X.isnull().sum().sum()
    print(f"   - Total missing values: {missing_count}")
    if missing_count == 0:
        print("   ✓ No missing values found!")
    
    # 4. Display summary statistics
    print(f"\n4. Summary Statistics:")
    print(X.describe().round(2))
    
    # 5. Target class distribution
    print(f"\n5. Target Class Distribution:")
    class_counts = y.value_counts()
    for class_idx, count in class_counts.items():
        percentage = (count / len(y)) * 100
        class_name = target_names[class_idx]
        print(f"   - {class_name}: {count} samples ({percentage:.1f}%)")
    
    # 6. Create correlation heatmap
    print(f"\n6. Generating Correlation Heatmap...")
    
    # Combine features and target for correlation analysis
    data_with_target = X.copy()
    data_with_target['target'] = y
    
    # Calculate correlation matrix
    correlation_matrix = data_with_target.corr()
    
    # Create figure with proper size
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Cancer Dataset Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("   ✓ Correlation heatmap saved as 'correlation_heatmap.png'")
    plt.close()
    
    # 7. Visualize target class distribution
    print(f"\n7. Generating Target Class Distribution Plot...")
    plt.figure(figsize=(10, 6))
    
    class_names = [target_names[i] for i in range(len(target_names))]
    colors = ['#FF6B6B', '#4ECDC4']
    
    counts = [len(y[y == i]) for i in range(len(target_names))]
    plt.bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    plt.title('Target Class Distribution (Benign vs Malignant)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✓ Class distribution plot saved as 'class_distribution.png'")
    plt.close()


def preprocess_data(X, y):
    """
    Preprocess the dataset: separate features/target, split data, and standardize features.
    
    Args:
        X (DataFrame): Feature data
        y (Series): Target data
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    # 1. Separate features and target (already done, but confirm)
    print(f"\n1. Features and Target Separated:")
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target shape: {y.shape}")
    
    # 2. Split dataset into training and testing sets (80:20 split)
    print(f"\n2. Splitting Dataset (80:20 train-test split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   - Training set size: {X_train.shape[0]} samples")
    print(f"   - Testing set size: {X_test.shape[0]} samples")
    
    # 3. Standardize features using StandardScaler
    print(f"\n3. Standardizing Features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ Features standardized successfully!")
    print(f"   - Mean of scaled training data: {X_train_scaled.mean():.6f}")
    print(f"   - Std of scaled training data: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple machine learning models on the dataset.
    
    Args:
        X_train (array): Scaled training features
        X_test (array): Scaled testing features
        y_train (Series): Training target
        y_test (Series): Testing target
    
    Returns:
        dict: Dictionary containing trained models and their performance metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)
    
    # Dictionary to store all models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        print(f"  ✓ {model_name} trained!")
        print(f"    - Accuracy: {accuracy:.4f}")
        print(f"    - Precision: {precision:.4f}")
        print(f"    - Recall: {recall:.4f}")
        print(f"    - F1 Score: {f1:.4f}")
    
    return results


def evaluate_and_compare_models(results):
    """
    Compare and display model performance metrics in a table format.
    
    Args:
        results (dict): Dictionary containing model results and metrics
    
    Returns:
        str: Name of the best-performing model
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON AND PERFORMANCE EVALUATION")
    print("=" * 70)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy (descending)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Display comparison table
    print("\n📊 MODEL PERFORMANCE COMPARISON TABLE:")
    print("-" * 90)
    print(comparison_df.to_string(index=False))
    print("-" * 90)
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    
    # Display confusion matrices for top 3 models
    print(f"\n🔍 CONFUSION MATRICES (Top 3 Models):")
    print("-" * 90)
    for idx, row in comparison_df.head(3).iterrows():
        model_name = row['Model']
        conf_matrix = results[model_name]['confusion_matrix']
        print(f"\n{model_name}:")
        print(f"  True Negatives:  {conf_matrix[0][0]:<5} | False Positives: {conf_matrix[0][1]}")
        print(f"  False Negatives: {conf_matrix[1][0]:<5} | True Positives:  {conf_matrix[1][1]}")
    
    return best_model_name


def save_best_model(results, best_model_name):
    """
    Save the best-performing model to a pickle file.
    
    Args:
        results (dict): Dictionary containing model results
        best_model_name (str): Name of the best model
    """
    print("\n" + "=" * 70)
    print("SAVING BEST MODEL")
    print("=" * 70)
    
    best_model = results[best_model_name]['model']
    filename = 'best_cancer_model.pkl'
    
    try:
        with open(filename, 'wb') as file:
            pickle.dump(best_model, file)
        print(f"\n✓ Best model saved successfully!")
        print(f"  - Model: {best_model_name}")
        print(f"  - File: {filename}")
        print(f"  - Accuracy: {results[best_model_name]['accuracy']:.4f}")
    except Exception as e:
        print(f"\n✗ Error saving model: {str(e)}")


def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    try:
        # Load and prepare data
        X, y, feature_names, target_names = load_and_prepare_data()
        
        # Perform EDA
        exploratory_data_analysis(X, y, target_names)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # Train models
        results = train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate and compare models
        best_model_name = evaluate_and_compare_models(results)
        
        # Save best model
        save_best_model(results, best_model_name)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Run 'python predict.py' for command-line predictions")
        print("2. Run 'python app.py' to launch the Flask web application")
        print("3. Open http://localhost:5000 in your browser")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
