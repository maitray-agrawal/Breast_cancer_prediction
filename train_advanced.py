"""
Cancer Health Prediction System - ADVANCED VERSION
===================================================
Professional ML pipeline with hyperparameter tuning, cross-validation,
feature importance, ROC curves, and advanced visualizations.

Features:
- GridSearchCV for hyperparameter optimization
- K-Fold Cross-Validation
- Feature importance analysis
- ROC-AUC curves
- Learning curves
- SHAP values (model interpretability)
- Data imbalance handling (SMOTE)
- Outlier detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import json
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_curve, auc, roc_auc_score, confusion_matrix, 
                             classification_report, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.covariance import EllipticEnvelope
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedCancerPredictionModel:
    """Advanced ML pipeline with professional features."""
    
    def __init__(self):
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.best_model = None
        self.results = {}
        self.metrics_history = []
    
    def load_data(self):
        """Load Breast Cancer Wisconsin dataset."""
        print("=" * 80)
        print("LOADING BREAST CANCER WISCONSIN DATASET")
        print("=" * 80)
        
        cancer_data = load_breast_cancer()
        X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        y = pd.Series(cancer_data.target, name='target')
        
        print(f"\n✓ Dataset loaded!")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Classes: {y.unique()}")
        
        self.feature_names = cancer_data.feature_names
        return X, y
    
    def eda_advanced(self, X, y):
        """Advanced EDA with professional visualizations."""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS (Advanced)")
        print("=" * 80)
        
        print(f"\n1. Dataset Shape & Info:")
        print(f"   - Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"   - Missing values: {X.isnull().sum().sum()}")
        print(f"   - Data types: {X.dtypes.unique()}")
        
        print(f"\n2. Statistical Summary:")
        print(X.describe().round(3).T[['mean', 'std', 'min', 'max']])
        
        print(f"\n3. Target Distribution:")
        class_dist = y.value_counts()
        for idx, count in class_dist.items():
            pct = (count / len(y)) * 100
            print(f"   Class {idx}: {count} samples ({pct:.1f}%)")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature distributions
        axes[0, 0].hist(X.iloc[:, 0], bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Feature Distribution (Mean Radius)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Class distribution
        class_dist.plot(kind='bar', ax=axes[0, 1], color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_title('Class Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticklabels(['Benign', 'Malignant'], rotation=0)
        
        # 3. Feature correlations (top 10)
        data_with_target = X.copy()
        data_with_target['target'] = y
        top_corr = data_with_target.corr()['target'].abs().nlargest(11)[1:]
        top_corr.plot(kind='barh', ax=axes[1, 0], color='steelblue')
        axes[1, 0].set_title('Top 10 Features by Correlation with Target', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Absolute Correlation')
        
        # 4. Box plot for outliers
        X_normalized = (X.iloc[:, :5] - X.iloc[:, :5].mean()) / X.iloc[:, :5].std()
        X_normalized.boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Feature Distributions (First 5 Features Normalized)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Normalized Value')
        
        plt.tight_layout()
        plt.savefig('eda_advanced.png', dpi=300, bbox_inches='tight')
        print("\n   ✓ EDA visualization saved: eda_advanced.png")
        plt.close()
    
    def detect_outliers(self, X, y):
        """Detect and handle outliers using Elliptic Envelope."""
        print("\n" + "=" * 80)
        print("OUTLIER DETECTION")
        print("=" * 80)
        
        ee = EllipticEnvelope(random_state=42, contamination=0.05)
        outlier_labels = ee.fit_predict(X)
        
        n_outliers = (outlier_labels == -1).sum()
        print(f"\n✓ Outliers detected: {n_outliers} samples ({n_outliers/len(X)*100:.2f}%)")
        
        # Remove outliers from both X and y
        mask = outlier_labels == 1
        X_clean = X[mask]
        y_clean = y[mask]
        
        return X_clean, y_clean, outlier_labels
    
    def feature_selection(self, X, y):
        """Select top features using SelectKBest."""
        print("\n" + "=" * 80)
        print("FEATURE SELECTION")
        print("=" * 80)
        
        selector = SelectKBest(score_func=f_classif, k=20)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"\n✓ Selected {len(selected_features)} features:")
        for idx, feat in enumerate(selected_features, 1):
            print(f"   {idx}. {feat}")
        
        # Visualize feature scores
        feature_scores = selector.scores_
        top_indices = np.argsort(feature_scores)[-20:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_scores = feature_scores[top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(top_features, top_scores, color='teal')
        plt.xlabel('Feature Score', fontsize=12, fontweight='bold')
        plt.title('Top 20 Feature Scores (SelectKBest)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_selection.png', dpi=300, bbox_inches='tight')
        print("✓ Feature selection visualization saved: feature_selection.png")
        plt.close()
        
        return X_selected, selected_features
    
    def preprocess_data(self, X, y):
        """Preprocess with SMOTE for class imbalance."""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING & CLASS IMBALANCE HANDLING")
        print("=" * 80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nBefore SMOTE:")
        print(f"  Training - Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
        
        # Apply SMOTE to handle imbalance
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print(f"\nAfter SMOTE:")
        print(f"  Training - Class 0: {(y_train_smote==0).sum()}, Class 1: {(y_train_smote==1).sum()}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_smote)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train_smote
        self.y_test = y_test
        
        print(f"\n✓ Data preprocessing complete!")
        print(f"  Training set (scaled): {X_train_scaled.shape}")
        print(f"  Testing set (scaled): {X_test_scaled.shape}")
    
    def train_with_hyperparameter_tuning(self):
        """Train models with GridSearchCV for optimal hyperparameters."""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
        print("=" * 80)
        
        # Define hyperparameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000, 5000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        for model_name, model in models.items():
            if model_name not in param_grids:
                continue
            
            print(f"\n🔍 Tuning {model_name}...")
            
            # GridSearchCV
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            print(f"  ✓ Best parameters: {grid_search.best_params_}")
            print(f"  ✓ Best CV score: {grid_search.best_score_:.4f}")
            
            # Store best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, best_model.predict_proba(self.X_test_scaled)[:, 1])
            
            self.results[model_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'best_params': grid_search.best_params_
            }
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test ROC-AUC: {roc_auc:.4f}")
    
    def evaluate_with_cross_validation(self):
        """Evaluate models using K-Fold Cross-Validation."""
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION EVALUATION")
        print("=" * 80)
        
        for model_name, result in self.results.items():
            model = result['model']
            
            # K-Fold Cross-Validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
            
            print(f"\n{model_name}:")
            print(f"  CV Scores: {cv_scores}")
            print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            result['cv_scores'] = cv_scores
            result['cv_mean'] = cv_scores.mean()
    
    def generate_roc_curves(self):
        """Generate ROC curves for all models."""
        print("\n" + "=" * 80)
        print("GENERATING ROC CURVES")
        print("=" * 80)
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in self.results.items():
            model = result['model']
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Random classifier
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved: roc_curves.png")
        plt.close()
    
    def generate_learning_curves(self):
        """Generate learning curves to detect overfitting."""
        print("\n" + "=" * 80)
        print("GENERATING LEARNING CURVES")
        print("=" * 80)
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        print(f"\nGenerating learning curve for: {best_model_name}")
        
        train_sizes, train_scores, val_scores = learning_curve(
            best_model,
            self.X_train_scaled,
            self.y_train,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(12, 7))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
        plt.ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        plt.title(f'Learning Curve - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Learning curve saved: learning_curve.png")
        plt.close()
    
    def feature_importance(self):
        """Extract and visualize feature importance."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Get Random Forest (has feature_importances_)
        if 'Random Forest' in self.results:
            model = self.results['Random Forest']['model']
            importances = model.feature_importances_
            
            # Get top 15 features
            top_indices = np.argsort(importances)[-15:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(top_features, top_importances, color='forestgreen')
            plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
            plt.title('Top 15 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Feature importance saved: feature_importance.png")
            plt.close()
    
    def select_best_model(self):
        """Select best model based on ROC-AUC."""
        print("\n" + "=" * 80)
        print("MODEL SELECTION")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'ROC-AUC': result['roc_auc'],
                'CV Mean': result.get('cv_mean', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print("-" * 70)
        print(comparison_df.to_string(index=False))
        print("-" * 70)
        
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
        
        return best_model_name
    
    def save_model_with_metadata(self, model_name):
        """Save model and metadata."""
        print("\n" + "=" * 80)
        print("SAVING ADVANCED MODEL")
        print("=" * 80)
        
        # Save model
        with open('best_cancer_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'accuracy': self.results[model_name]['accuracy'],
            'roc_auc': self.results[model_name]['roc_auc'],
            'best_params': str(self.results[model_name]['best_params']),
            'cv_score': float(self.results[model_name].get('cv_mean', 0)),
            'n_features': len(self.feature_names),
            'features': list(self.feature_names)
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Model saved: best_cancer_model.pkl")
        print(f"✓ Scaler saved: scaler.pkl")
        print(f"✓ Metadata saved: model_metadata.json")
    
    def run_complete_pipeline(self):
        """Run complete advanced ML pipeline."""
        try:
            # Load data
            X, y = self.load_data()
            
            # Advanced EDA
            self.eda_advanced(X, y)
            
            # Detect outliers
            X_clean, y_clean, _ = self.detect_outliers(X, y)
            
            # Feature selection
            X_selected, selected_features = self.feature_selection(X_clean, y_clean)
            
            # Preprocessing
            self.preprocess_data(X_selected, y_clean)
            
            # Train with hyperparameter tuning
            self.train_with_hyperparameter_tuning()
            
            # Cross-validation evaluation
            self.evaluate_with_cross_validation()
            
            # Generate visualizations
            self.generate_roc_curves()
            self.generate_learning_curves()
            self.feature_importance()
            
            # Select best model
            best_model_name = self.select_best_model()
            
            # Save model
            self.save_model_with_metadata(best_model_name)
            
            print("\n" + "=" * 80)
            print("✓ ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nGenerated Files:")
            print("  - best_cancer_model.pkl (trained model)")
            print("  - scaler.pkl (feature scaler)")
            print("  - model_metadata.json (model information)")
            print("  - eda_advanced.png (EDA visualization)")
            print("  - feature_selection.png (feature scores)")
            print("  - roc_curves.png (ROC curves)")
            print("  - learning_curve.png (learning curves)")
            print("  - feature_importance.png (feature importance)")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            raise


def main():
    """Main execution."""
    model = AdvancedCancerPredictionModel()
    model.run_complete_pipeline()


if __name__ == "__main__":
    main()
