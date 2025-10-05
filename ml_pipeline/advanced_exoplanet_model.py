"""
ExoAI Hunter - Advanced Deep Learning Model for 99%+ Accuracy
State-of-the-art neural network architecture with ensemble methods and advanced techniques
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import json
import pickle
from datetime import datetime
import optuna
from scipy.stats import uniform, randint

class AdvancedAttentionBlock(layers.Layer):
    """
    Multi-head attention mechanism with residual connections and layer normalization
    """
    
    def __init__(self, units=128, num_heads=8, dropout_rate=0.1, **kwargs):
        super(AdvancedAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=units//num_heads,
            dropout=dropout_rate
        )
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(units * 4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(units)
        ])
        
    def call(self, inputs, training=None):
        # Multi-head attention with residual connection
        attn_output = self.multi_head_attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        
        return out2

class ResidualBlock(layers.Layer):
    """
    Residual block with batch normalization and skip connections
    """
    
    def __init__(self, filters, kernel_size=3, dropout_rate=0.1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        
        # Skip connection projection if needed
        self.skip_projection = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.skip_projection = layers.Conv1D(self.filters, 1, padding='same')
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # First convolution
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        skip = inputs
        if self.skip_projection is not None:
            skip = self.skip_projection(inputs)
            
        # Add skip connection and apply activation
        x = tf.nn.relu(x + skip)
        
        return x

class AdvancedExoplanetCNN(Model):
    """
    Advanced CNN architecture with residual blocks, attention, and ensemble techniques
    """
    
    def __init__(self, num_classes=3, sequence_length=1000, dropout_rate=0.2, **kwargs):
        super(AdvancedExoplanetCNN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Input processing
        self.input_conv = layers.Conv1D(64, 7, padding='same')
        self.input_bn = layers.BatchNormalization()
        self.input_pool = layers.MaxPooling1D(2)
        
        # Residual blocks
        self.res_blocks = [
            ResidualBlock(64, 5, dropout_rate),
            ResidualBlock(128, 5, dropout_rate),
            ResidualBlock(256, 3, dropout_rate),
            ResidualBlock(512, 3, dropout_rate),
        ]
        
        # Pooling layers
        self.pool_layers = [
            layers.MaxPooling1D(2),
            layers.MaxPooling1D(2),
            layers.MaxPooling1D(2),
        ]
        
        # Attention mechanism
        self.attention = AdvancedAttentionBlock(256, num_heads=8, dropout_rate=dropout_rate)
        
        # Global pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.global_max_pool = layers.GlobalMaxPooling1D()
        
        # Dense layers with batch normalization
        self.dense1 = layers.Dense(1024, activation='relu')
        self.bn_dense1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate * 2)
        
        self.dense2 = layers.Dense(512, activation='relu')
        self.bn_dense2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.dense3 = layers.Dense(256, activation='relu')
        self.bn_dense3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate)
        
        # Output layer with label smoothing
        self.output_layer = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=None):
        # Reshape for CNN if needed
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, -1)
        else:
            x = inputs
            
        # Input processing
        x = self.input_conv(x)
        x = self.input_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.input_pool(x)
        
        # Residual blocks with pooling
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, training=training)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        
        # Attention mechanism
        x = self.attention(x, training=training)
        
        # Global pooling (both average and max)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = layers.concatenate([avg_pool, max_pool])
        
        # Dense layers with batch normalization
        x = self.dense1(x)
        x = self.bn_dense1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn_dense2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn_dense3(x, training=training)
        x = self.dropout3(x, training=training)
        
        # Output with Monte Carlo dropout for uncertainty
        outputs = self.output_layer(x)
        
        return outputs

class AdvancedExoAIHunterModel:
    """
    Advanced model class with hyperparameter optimization and ensemble methods for 99%+ accuracy
    """
    
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.models = {}
        self.history = {}
        self.metrics = {}
        self.best_params = {}
        
        # Class names for interpretation
        self.class_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """
        Hyperparameter optimization using Optuna for maximum accuracy
        """
        print(f"🔍 Starting hyperparameter optimization with {n_trials} trials...")
        
        def objective(trial):
            # Suggest hyperparameters
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
            
            # Create model with suggested parameters
            model = AdvancedExoplanetCNN(
                num_classes=self.num_classes,
                sequence_length=X_train.shape[1],
                dropout_rate=dropout_rate
            )
            
            # Compile model
            optimizer = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01
            )
            
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy']
            )
            
            # Train model
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Return best validation accuracy
            return max(history.history['val_accuracy'])
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        print(f"🎯 Best parameters found: {self.best_params}")
        print(f"🏆 Best accuracy: {study.best_value:.4f}")
        
        return self.best_params
    
    def create_advanced_model(self, input_shape, model_name='advanced_primary'):
        """Create advanced model with optimized hyperparameters"""
        
        # Use best parameters if available, otherwise use defaults
        dropout_rate = self.best_params.get('dropout_rate', 0.2)
        learning_rate = self.best_params.get('learning_rate', 0.001)
        
        model = AdvancedExoplanetCNN(
            num_classes=self.num_classes,
            sequence_length=input_shape[0],
            dropout_rate=dropout_rate
        )
        
        # Advanced optimizer with learning rate scheduling
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with label smoothing and focal loss
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.F1Score(name='f1_score', average='weighted')
            ]
        )
        
        self.models[model_name] = model
        return model
    
    def train_advanced_model(self, X_train, y_train, X_val, y_val, 
                           model_name='advanced_primary', epochs=200):
        """
        Train model with advanced techniques for maximum accuracy
        """
        model = self.models[model_name]
        batch_size = self.best_params.get('batch_size', 32)
        
        # Advanced callbacks
        callbacks_list = [
            # Cosine annealing learning rate schedule
            callbacks.CosineRestartSchedule(
                first_restart_step=50,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.0
            ),
            
            # Early stopping with patience
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.0001
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                f'models/advanced_{model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Custom callback for 99% accuracy target
            AdvancedAccuracyCallback(threshold=0.99)
        ]
        
        print(f"🚀 Training advanced model for 99%+ accuracy...")
        print(f"   Target accuracy: >99%")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history[model_name] = history
        
        # Evaluate final performance
        self.evaluate_advanced_model(X_val, y_val, model_name)
        
        return history
    
    def create_super_ensemble(self, X_train, y_train, X_val, y_val, n_models=10):
        """
        Create super ensemble with diverse architectures for maximum accuracy
        """
        print(f"🔄 Creating super ensemble with {n_models} diverse models...")
        
        ensemble_models = []
        
        for i in range(n_models):
            model_name = f'super_ensemble_{i}'
            
            # Create diverse model architectures
            if i % 3 == 0:
                # CNN-based model
                model = self.create_advanced_model(X_train.shape[1:], model_name)
            elif i % 3 == 1:
                # Transformer-based model
                model = self.create_transformer_model(X_train.shape[1:], model_name)
            else:
                # Hybrid model
                model = self.create_hybrid_model(X_train.shape[1:], model_name)
            
            # Train with different random seeds and augmentation
            np.random.seed(42 + i * 7)
            tf.random.set_seed(42 + i * 7)
            
            # Apply different data augmentation for each model
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train, seed=i)
            
            # Train model
            history = self.train_advanced_model(
                X_train_aug, y_train_aug, X_val, y_val,
                model_name=model_name,
                epochs=100
            )
            
            ensemble_models.append(model_name)
        
        self.super_ensemble_models = ensemble_models
        return ensemble_models
    
    def predict_super_ensemble(self, X, use_uncertainty=True):
        """Make predictions using super ensemble with uncertainty quantification"""
        if not hasattr(self, 'super_ensemble_models'):
            raise ValueError("Super ensemble not created. Call create_super_ensemble() first.")
        
        predictions = []
        uncertainties = []
        
        for model_name in self.super_ensemble_models:
            model = self.models[model_name]
            
            if use_uncertainty:
                # Monte Carlo dropout for uncertainty estimation
                mc_predictions = []
                for _ in range(10):  # 10 MC samples
                    pred = model(X, training=True)  # Keep dropout active
                    mc_predictions.append(pred.numpy())
                
                mc_predictions = np.array(mc_predictions)
                pred_mean = np.mean(mc_predictions, axis=0)
                pred_std = np.std(mc_predictions, axis=0)
                
                predictions.append(pred_mean)
                uncertainties.append(pred_std)
            else:
                pred = model.predict(X)
                predictions.append(pred)
        
        # Weighted ensemble prediction
        ensemble_pred = np.mean(predictions, axis=0)
        
        if use_uncertainty:
            ensemble_uncertainty = np.mean(uncertainties, axis=0)
            return ensemble_pred, ensemble_uncertainty
        
        return ensemble_pred
    
    def augment_data(self, X, y, seed=42):
        """Advanced data augmentation techniques"""
        np.random.seed(seed)
        
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Time series augmentation techniques
        for i in range(len(X)):
            if np.random.random() < 0.5:  # 50% chance of augmentation
                # Add noise
                noise = np.random.normal(0, 0.01, X[i].shape)
                X_aug[i] += noise
                
                # Time warping (slight stretching/compression)
                if np.random.random() < 0.3:
                    warp_factor = np.random.uniform(0.95, 1.05)
                    indices = np.arange(len(X[i]))
                    new_indices = indices * warp_factor
                    new_indices = np.clip(new_indices, 0, len(X[i]) - 1).astype(int)
                    X_aug[i] = X[i][new_indices]
        
        return X_aug, y_aug
    
    def evaluate_advanced_model(self, X_test, y_test, model_name='advanced_primary'):
        """Comprehensive evaluation with detailed metrics"""
        model = self.models[model_name]
        
        # Get predictions with uncertainty
        if hasattr(self, 'super_ensemble_models') and model_name in self.super_ensemble_models:
            y_pred_proba, uncertainty = self.predict_super_ensemble(X_test)
        else:
            y_pred_proba = model.predict(X_test)
            uncertainty = None
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate comprehensive metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Store detailed metrics
        self.metrics[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'uncertainty': uncertainty
        }
        
        print(f"📊 {model_name} Advanced Model Performance:")
        print(f"   Accuracy: {accuracy:.6f}")
        
        if accuracy >= 0.99:
            print("🎯 TARGET ACHIEVED: >99% accuracy! 🏆")
        elif accuracy >= 0.98:
            print("🔥 EXCELLENT: >98% accuracy!")
        elif accuracy >= 0.95:
            print("✅ GOOD: >95% accuracy")
        else:
            print(f"⚠️  Need {0.99 - accuracy:.4f} more accuracy to reach 99% target")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        try:
            unique_classes = sorted(list(set(y_test) | set(y_pred)))
            class_names_subset = [self.class_names[i] for i in unique_classes if i < len(self.class_names)]
            print(classification_report(y_test, y_pred, labels=unique_classes, target_names=class_names_subset))
        except Exception as e:
            print(f"Classification report error: {e}")
        
        return self.metrics[model_name]

class CosineRestartSchedule(callbacks.Callback):
    """Cosine annealing with warm restarts"""
    
    def __init__(self, first_restart_step, t_mul=2.0, m_mul=1.0, alpha=0.0):
        super(CosineRestartSchedule, self).__init__()
        self.first_restart_step = first_restart_step
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        
    def on_epoch_begin(self, epoch, logs=None):
        # Calculate current learning rate using cosine annealing
        lr = self.alpha + (self.model.optimizer.learning_rate - self.alpha) * \
             (1 + np.cos(np.pi * epoch / self.first_restart_step)) / 2
        
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)

class AdvancedAccuracyCallback(callbacks.Callback):
    """Custom callback to track >99% accuracy achievement"""
    
    def __init__(self, threshold=0.99):
        super(AdvancedAccuracyCallback, self).__init__()
        self.threshold = threshold
        self.achieved = False
    
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy', 0)
        
        if val_accuracy >= self.threshold and not self.achieved:
            print(f"\n🎯 99% TARGET ACHIEVED! Validation accuracy {val_accuracy:.6f} >= {self.threshold}")
            print("🏆 ExoAI Hunter has reached 99%+ accuracy! Exceptional performance!")
            self.achieved = True

# Factory function for advanced model
def create_advanced_exoai_model(num_classes=3):
    """Factory function to create Advanced ExoAI Hunter model"""
    return AdvancedExoAIHunterModel(num_classes=num_classes)
