"""
SageMaker-Compatible Training Script
Trains customer churn prediction model
"""

import argparse
import json
import logging
import os
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from churn_model import ChurnPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './models/artifacts'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/processed'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data/processed'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--config-path', type=str, default='config/config.yaml')
    
    return parser.parse_args()


def load_data(data_dir: str, target_col: str = 'churn'):
    """Load training/validation data"""
    logger.info(f"Loading data from {data_dir}")
    
    # Find CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Load data
    df = pd.read_csv(os.path.join(data_dir, csv_files[0]))
    logger.info(f"Loaded {len(df)} records")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    y = df[target_col].values
    X = df.drop(columns=[target_col, 'customer_id'], errors='ignore').values
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def save_model_artifacts(model, model_dir: str, metrics: dict):
    """Save model and metrics"""
    logger.info(f"Saving model to {model_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model in TensorFlow SavedModel format (SageMaker compatible)
    model_path = os.path.join(model_dir, '1')  # Version 1
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'tensorflow',
        'framework_version': tf.__version__,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def train():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting training job...")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    if os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config for SageMaker
        config = {
            'model': {
                'architecture': {},
                'training': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'early_stopping_patience': 5
                },
                'performance': {
                    'min_accuracy': 0.84,
                    'min_f1_score': 0.79
                }
            }
        }
    
    # Update config with command-line arguments
    config['model']['training']['epochs'] = args.epochs
    config['model']['training']['batch_size'] = args.batch_size
    config['model']['training']['learning_rate'] = args.learning_rate
    
    # Load training and validation data
    X_train, y_train = load_data(args.train)
    
    if os.path.exists(args.validation):
        X_val, y_val = load_data(args.validation)
    else:
        # Split training data if no validation set provided
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        logger.info("Created validation split from training data")
    
    # Initialize model
    churn_model = ChurnPredictionModel(config)
    
    # Build and train model
    model = churn_model.build_model(input_dim=X_train.shape[1])
    history = churn_model.train(
        X_train, y_train,
        X_val, y_val,
        model_dir=args.model_dir
    )
    
    # Evaluate on validation set
    logger.info("Evaluating model on validation set...")
    val_loss, val_acc, val_precision, val_recall, val_auc, val_f1 = model.evaluate(
        X_val, y_val,
        verbose=0
    )
    
    # Get predictions for detailed metrics
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'validation_loss': float(val_loss),
        'validation_accuracy': float(val_acc),
        'validation_precision': float(val_precision),
        'validation_recall': float(val_recall),
        'validation_auc': float(val_auc),
        'validation_f1_score': float(val_f1)
    }
    
    logger.info(f"Validation Metrics: {metrics}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Check performance thresholds
    min_acc = config['model']['performance']['min_accuracy']
    min_f1 = config['model']['performance']['min_f1_score']
    
    if val_acc >= min_acc and val_f1 >= min_f1:
        logger.info(f"✅ Model meets performance thresholds!")
        logger.info(f"   Accuracy: {val_acc:.4f} >= {min_acc}")
        logger.info(f"   F1-Score: {val_f1:.4f} >= {min_f1}")
        metrics['meets_thresholds'] = True
    else:
        logger.warning(f"⚠️ Model does not meet performance thresholds")
        logger.warning(f"   Accuracy: {val_acc:.4f} < {min_acc}")
        logger.warning(f"   F1-Score: {val_f1:.4f} < {min_f1}")
        metrics['meets_thresholds'] = False
    
    # Save model and artifacts
    save_model_artifacts(churn_model.model, args.model_dir, metrics)
    
    logger.info("Training completed successfully!")
    
    return metrics


if __name__ == "__main__":
    train()
