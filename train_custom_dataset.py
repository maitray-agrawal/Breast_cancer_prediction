"""
Cancer Health Prediction System - Train with Custom Dataset
============================================================
This script allows you to train the model with custom datasets or 
the built-in Breast Cancer Wisconsin dataset.

Supports: CSV files, Excel files, or sklearn built-in datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_builtin_dataset():
    """Load the built-in Breast Cancer Wisconsin dataset."""
    print("=" * 70)
    print("LOADING BREAST CANCER WISCONSIN DATASET (Built-in)")
    print("=" * 70)
    
    cancer_data = load_breast_cancer()
    X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y = pd.Series(cancer_data.target, name='target')
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Features: {X.shape[1]}")
    
    return X, y, cancer_data.feature_names, cancer_data.target_names


def load_csv_dataset(filepath, target_column=None):
    """
    Load a custom dataset from CSV file.
    
    Args:
        filepath (str): Path to CSV file
        target_column (str): Name of the target column
                            If None, last column is used as target
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    print("=" * 70)
    print(f"LOADING CUSTOM DATASET FROM CSV: {filepath}")
    print("=" * 70)
    
    try:
        # Load CSV file
        df = pd.read_csv(filepath)
        print(f"\n✓ CSV file loaded successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Identify target column
        if target_column is None:
            # Use last column as target
            target_column = df.columns[-1]
            print(f"  - Using last column as target: {target_column}")
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in CSV")
            print(f"  - Target column: {target_column}")
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Get unique target classes
        target_names = np.unique(y)
        
        # Convert target to numeric if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name='target')
            target_names = le.classes_
            print(f"  - Target classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        print(f"  - Features: {len(feature_names)}")
        print(f"  - Target classes: {list(target_names)}")
        
        return X, y, feature_names, target_names
    
    except FileNotFoundError:
        print(f"\n✗ Error: File '{filepath}' not found!")
        print(f"  - Please place your CSV file in: {os.path.abspath(filepath)}")
        raise
    except Exception as e:
        print(f"\n✗ Error loading CSV: {str(e)}")
        raise


def load_excel_dataset(filepath, sheet_name=0, target_column=None):
    """
    Load a custom dataset from Excel file.
    
    Args:
        filepath (str): Path to Excel file (.xlsx, .xls)
        sheet_name (str/int): Sheet name or index
        target_column (str): Name of the target column
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    print("=" * 70)
    print(f"LOADING CUSTOM DATASET FROM EXCEL: {filepath}")
    print("=" * 70)
    
    try:
        # Load Excel file
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"\n✓ Excel file loaded successfully!")
        print(f"  - Sheet: {sheet_name}")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Identify target column
        if target_column is None:
            target_column = df.columns[-1]
            print(f"  - Using last column as target: {target_column}")
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            print(f"  - Target column: {target_column}")
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Get unique target classes
        target_names = np.unique(y)
        
        # Convert target to numeric if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name='target')
            target_names = le.classes_
        
        print(f"  - Features: {len(feature_names)}")
        print(f"  - Target classes: {list(target_names)}")
        
        return X, y, feature_names, target_names
    
    except FileNotFoundError:
        print(f"\n✗ Error: File '{filepath}' not found!")
        raise
    except Exception as e:
        print(f"\n✗ Error loading Excel: {str(e)}")
        raise


def load_dataset(dataset_source='builtin', filepath=None, target_column=None):
    """
    Load dataset based on source.
    
    Args:
        dataset_source (str): 'builtin', 'csv', or 'excel'
        filepath (str): Path to file (for csv/excel)
        target_column (str): Name of target column
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    if dataset_source == 'builtin':
        return load_builtin_dataset()
    elif dataset_source == 'csv':
        if filepath is None:
            raise ValueError("filepath required for CSV dataset")
        return load_csv_dataset(filepath, target_column)
    elif dataset_source == 'excel':
        if filepath is None:
            raise ValueError("filepath required for Excel dataset")
        return load_excel_dataset(filepath, target_column=target_column)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_source}")


def exploratory_data_analysis(X, y, target_names):
    """Perform EDA on the dataset."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    print(f"\n1. Dataset Shape:")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    
    print(f"\n2. Feature Names ({X.shape[1]} total):")
    for idx, col in enumerate(X.columns[:10], 1):  # Show first 10
        print(f"   {idx}. {col}")
    if X.shape[1] > 10:
        print(f"   ... and {X.shape[1] - 10} more features")
    
    print(f"\n3. Missing Values:")
    missing_count = X.isnull().sum().sum()
    print(f"   - Total missing values: {missing_count}")
    if missing_count > 0:
        # Handle missing values
        print("   - Filling missing values with mean...")
        X = X.fillna(X.mean())
    else:
        print("   ✓ No missing values found!")
    
    print(f"\n4. Summary Statistics:")
    print(X.describe().round(2))
    
    print(f"\n5. Target Class Distribution:")
    class_counts = y.value_counts()
    for class_idx, count in class_counts.items():
        percentage = (count / len(y)) * 100
        class_name = target_names[class_idx] if class_idx < len(target_names) else str(class_idx)
        print(f"   - {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Generate visualizations
    try:
        print(f"\n6. Generating visualizations...")
        
        # Correlation heatmap
        plt.figure(figsize=(14, 12))
        data_with_target = X.copy()
        data_with_target['target'] = y
        correlation_matrix = data_with_target.corr()
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("   ✓ Correlation heatmap saved")
        plt.close()
        
        # Class distribution
        plt.figure(figsize=(10, 6))
        counts = [len(y[y == i]) for i in range(len(target_names))]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(target_names)]
        plt.bar(range(len(target_names)), counts, color=colors, alpha=0.7, edgecolor='black')
        plt.xticks(range(len(target_names)), target_names)
        plt.title('Target Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples')
        for i, v in enumerate(counts):
            plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("   ✓ Class distribution plot saved")
        plt.close()
    
    except Exception as e:
        print(f"   ⚠ Could not generate visualizations: {str(e)}")
    
    return X


def preprocess_data(X, y):
    """Preprocess and split data."""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    print(f"\n1. Dataset shape confirmed: {X.shape}")
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    print(f"\n2. Splitting dataset (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   - Training: {X_train.shape[0]} samples")
    print(f"   - Testing: {X_test.shape[0]} samples")
    
    print(f"\n3. Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ Features standardized")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models."""
    print("\n" + "=" * 70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        print(f"  ✓ {model_name}")
        print(f"    - Accuracy: {accuracy:.4f}")
        print(f"    - Precision: {precision:.4f}")
        print(f"    - Recall: {recall:.4f}")
        print(f"    - F1 Score: {f1:.4f}")
    
    return results


def evaluate_and_compare_models(results):
    """Compare model performance."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON AND EVALUATION")
    print("=" * 70)
    
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
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print("\n📊 MODEL PERFORMANCE TABLE:")
    print("-" * 90)
    print(comparison_df.to_string(index=False))
    print("-" * 90)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    
    return best_model_name


def save_best_model(results, best_model_name):
    """Save the best model."""
    print("\n" + "=" * 70)
    print("SAVING BEST MODEL")
    print("=" * 70)
    
    best_model = results[best_model_name]['model']
    filename = 'best_cancer_model.pkl'
    
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    
    print(f"\n✓ Model saved: {filename}")
    print(f"  - Model: {best_model_name}")
    print(f"  - Accuracy: {results[best_model_name]['accuracy']:.4f}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("CANCER HEALTH PREDICTION SYSTEM - DATASET SELECTION")
    print("=" * 70)
    
    print("\n📊 SELECT DATASET SOURCE:")
    print("1. Built-in Breast Cancer Wisconsin Dataset (Recommended for testing)")
    print("2. Load CSV file from disk")
    print("3. Load Excel file from disk")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    try:
        if choice == '1':
            # Use built-in dataset
            X, y, feature_names, target_names = load_builtin_dataset()
        
        elif choice == '2':
            # Load CSV dataset
            filepath = input("Enter path to CSV file: ").strip()
            target_col = input("Enter target column name (press Enter to use last column): ").strip()
            target_col = target_col if target_col else None
            X, y, feature_names, target_names = load_csv_dataset(filepath, target_col)
        
        elif choice == '3':
            # Load Excel dataset
            filepath = input("Enter path to Excel file: ").strip()
            target_col = input("Enter target column name (press Enter to use last column): ").strip()
            target_col = target_col if target_col else None
            X, y, feature_names, target_names = load_excel_dataset(filepath, target_column=target_col)
        
        else:
            print("✗ Invalid choice!")
            return
        
        # Perform EDA
        X = exploratory_data_analysis(X, y, target_names)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # Train models
        results = train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        best_model_name = evaluate_and_compare_models(results)
        
        # Save best model
        save_best_model(results, best_model_name)
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. python predict.py - Command-line predictions")
        print("2. python app.py - Web interface (http://localhost:5000)")
        
    except KeyboardInterrupt:
        print("\n\n✗ Training cancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    main()
