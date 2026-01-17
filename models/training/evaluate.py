"""
Model Evaluation Script
Comprehensive evaluation of trained models
"""

import argparse
import json
import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained model performance"""
    
    def __init__(self, model_path: str):
        """Initialize evaluator"""
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Starting model evaluation...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        logger.info(f"Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        output_path: str = 'confusion_matrix.png'
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
        plt.close()
    
    def plot_roc_curve(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        output_path: str = 'roc_curve.png'
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"ROC curve saved to {output_path}")
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        output_path: str = 'precision_recall_curve.png'
    ):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Precision-Recall curve saved to {output_path}")
        plt.close()
    
    def generate_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: str = './evaluation_reports'
    ) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Output directory for reports
            
        Returns:
            Evaluation metrics
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = self.evaluate(X_test, y_test)
        
        # Generate visualizations
        self.plot_confusion_matrix(
            y_test, y_pred,
            os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        self.plot_roc_curve(
            y_test, y_pred_proba,
            os.path.join(output_dir, 'roc_curve.png')
        )
        
        self.plot_precision_recall_curve(
            y_test, y_pred_proba,
            os.path.join(output_dir, 'precision_recall_curve.png')
        )
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=['Not Churned', 'Churned'],
            output_dict=True
        )
        
        # Save detailed report
        report = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--test-data', required=True, help='Path to test data CSV')
    parser.add_argument('--output-dir', default='./evaluation_reports', help='Output directory')
    
    args = parser.parse_args()
    
    # Load test data
    df = pd.read_csv(args.test_data)
    y_test = df['churn'].values
    X_test = df.drop(columns=['churn', 'customer_id'], errors='ignore').values
    
    logger.info(f"Loaded test data: {X_test.shape}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path)
    
    # Generate report
    report = evaluator.generate_report(X_test, y_test, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    for metric, value in report['metrics'].items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*50)
    
    # Check thresholds
    if report['metrics']['accuracy'] >= 0.84 and report['metrics']['f1_score'] >= 0.79:
        print("✅ Model PASSES performance thresholds!")
        return 0
    else:
        print("⚠️ Model DOES NOT meet performance thresholds")
        return 1


if __name__ == "__main__":
    exit(main())
