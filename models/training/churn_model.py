"""
TensorFlow Customer Churn Prediction Model
Deep neural network architecture for binary classification
"""

import logging
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictionModel:
    """Customer churn prediction model using TensorFlow"""
    
    def __init__(self, config: Dict):
        """
        Initialize model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build TensorFlow model architecture
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model with input dimension: {input_dim}")
        
        # Model architecture from config
        arch_config = self.config['model']['architecture']
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='input_features')
        
        # First hidden layer
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_1'
        )(inputs)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        # Second hidden layer
        x = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Third hidden layer
        x = layers.Dense(
            32,
            activation='relu',
            name='dense_3'
        )(x)
        x = layers.Dropout(0.2, name='dropout_3')(x)
        
        # Output layer
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='churn_prediction_model')
        
        # Compile model
        learning_rate = self.config['model']['training']['learning_rate']
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                self._f1_score
            ]
        )
        
        self.model = model
        logger.info("Model built successfully")
        logger.info(model.summary())
        
        return model
    
    @staticmethod
    def _f1_score(y_true, y_pred):
        """Calculate F1 score"""
        precision = keras.metrics.Precision()
        recall = keras.metrics.Recall()
        
        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)
        
        p = precision.result()
        r = recall.result()
        
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
    
    def get_callbacks(self, model_dir: str) -> list:
        """
        Get training callbacks
        
        Args:
            model_dir: Directory to save model checkpoints
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['model']['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=f"{model_dir}/best_model.h5",
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=f"{model_dir}/logs",
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        model_dir: str = './models/artifacts'
    ) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_dir: Directory to save artifacts
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Calculate class weights for imbalanced data
        class_weights = self._calculate_class_weights(y_train)
        
        # Get callbacks
        callbacks = self.get_callbacks(model_dir)
        
        # Training configuration
        batch_size = self.config['model']['training']['batch_size']
        epochs = self.config['model']['training']['epochs']
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        logger.info("Training completed")
        
        return history
    
    def _calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights
    
    def save_model(self, model_path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save in TensorFlow SavedModel format
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save in H5 format for compatibility
        h5_path = model_path.replace('savedmodel', 'model.h5')
        self.model.save(h5_path)
        logger.info(f"Model also saved to {h5_path}")
    
    def load_model(self, model_path: str):
        """Load model from disk"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def predict(self, X, threshold: float = 0.5):
        """
        Make predictions
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        probabilities = self.model.predict(X)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities


def main():
    """Main execution for testing"""
    import yaml
    import numpy as np
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data for testing
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 2, 200)
    
    # Initialize and build model
    churn_model = ChurnPredictionModel(config)
    model = churn_model.build_model(input_dim=20)
    
    logger.info("Model architecture test successful")


if __name__ == "__main__":
    main()
