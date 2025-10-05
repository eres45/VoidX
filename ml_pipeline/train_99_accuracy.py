#!/usr/bin/env python3
"""
ExoAI Hunter - 99%+ Accuracy Training Script
Advanced training pipeline to achieve world-class exoplanet detection accuracy
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_exoplanet_model import AdvancedExoAIHunterModel
from data_processor import ExoplanetDataProcessor
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_dataset(n_samples=50000):
    """
    Create a large, high-quality synthetic dataset for training
    """
    print("🔬 Creating high-quality synthetic dataset...")
    
    np.random.seed(42)
    
    # Generate diverse light curves
    data = []
    labels = []
    
    for i in range(n_samples):
        # Random parameters
        length = np.random.randint(800, 1200)
        time = np.linspace(0, 100, length)
        
        # Class distribution: 40% confirmed, 35% candidate, 25% false positive
        class_prob = np.random.random()
        if class_prob < 0.4:
            # Confirmed exoplanet - strong, regular transits
            label = 0  # CONFIRMED
            period = np.random.uniform(1, 50)
            depth = np.random.uniform(0.01, 0.1)
            duration = np.random.uniform(0.5, 4.0)
            
            # Base stellar flux with variability
            flux = 1.0 + 0.005 * np.sin(0.1 * time) + 0.002 * np.random.normal(0, 1, length)
            
            # Add periodic transits
            for t in time:
                phase = (t % period) / period
                if 0.45 < phase < 0.55:  # Transit window
                    transit_shape = 1 - depth * np.exp(-((phase - 0.5) / (duration/period))**2)
                    flux[int(t * length / 100)] *= transit_shape
                    
        elif class_prob < 0.75:
            # Planet candidate - weaker or irregular signals
            label = 1  # CANDIDATE
            period = np.random.uniform(5, 100)
            depth = np.random.uniform(0.005, 0.05)
            duration = np.random.uniform(1.0, 6.0)
            
            # More stellar variability
            flux = 1.0 + 0.01 * np.sin(0.05 * time) + 0.008 * np.random.normal(0, 1, length)
            
            # Irregular or weak transits
            for t in time:
                if np.random.random() < 0.8:  # 80% chance of transit
                    phase = (t % period) / period
                    if 0.47 < phase < 0.53:
                        transit_shape = 1 - depth * np.exp(-((phase - 0.5) / (duration/period))**2)
                        flux[int(t * length / 100)] *= transit_shape
                        
        else:
            # False positive - stellar variability, eclipsing binaries, etc.
            label = 2  # FALSE_POSITIVE
            
            # Various false positive scenarios
            fp_type = np.random.choice(['stellar_var', 'eclipsing_binary', 'instrumental'])
            
            if fp_type == 'stellar_var':
                # Stellar variability
                flux = 1.0 + 0.02 * np.sin(0.03 * time) + 0.01 * np.sin(0.07 * time)
                flux += 0.015 * np.random.normal(0, 1, length)
                
            elif fp_type == 'eclipsing_binary':
                # Eclipsing binary system
                period = np.random.uniform(0.5, 10)
                depth1 = np.random.uniform(0.05, 0.3)  # Primary eclipse
                depth2 = np.random.uniform(0.01, 0.1)   # Secondary eclipse
                
                flux = np.ones(length)
                for t in time:
                    phase = (t % period) / period
                    if 0.48 < phase < 0.52:  # Primary eclipse
                        flux[int(t * length / 100)] *= (1 - depth1)
                    elif 0.98 < phase or phase < 0.02:  # Secondary eclipse
                        flux[int(t * length / 100)] *= (1 - depth2)
                        
                flux += 0.01 * np.random.normal(0, 1, length)
                
            else:  # instrumental
                # Instrumental artifacts
                flux = 1.0 + 0.005 * np.random.normal(0, 1, length)
                # Add systematic trends
                flux += 0.01 * np.linspace(-1, 1, length)
        
        # Normalize to standard length (1000 points)
        if length != 1000:
            flux_interp = np.interp(np.linspace(0, length-1, 1000), np.arange(length), flux)
            flux = flux_interp
        
        data.append(flux)
        labels.append(label)
    
    X = np.array(data)
    y = np.array(labels)
    
    print(f"✅ Created dataset with {n_samples} samples")
    print(f"   Class distribution: {np.bincount(y)}")
    
    return X, y

def train_99_accuracy_model():
    """
    Main training function to achieve 99%+ accuracy
    """
    print("🚀 Starting 99%+ Accuracy Training Pipeline")
    print("=" * 60)
    
    # Create high-quality dataset
    X, y = create_synthetic_dataset(n_samples=50000)
    
    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"📊 Dataset splits:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples") 
    print(f"   Test: {len(X_test)} samples")
    
    # Initialize advanced model
    model_trainer = AdvancedExoAIHunterModel(num_classes=3)
    
    # Hyperparameter optimization
    print("\n🔍 Optimizing hyperparameters...")
    best_params = model_trainer.optimize_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=30
    )
    
    # Create and train advanced model
    print("\n🏗️ Creating advanced model architecture...")
    input_shape = X_train.shape[1:]
    model = model_trainer.create_advanced_model(input_shape, 'advanced_99_model')
    
    # Train with advanced techniques
    print("\n🎯 Training for 99%+ accuracy...")
    history = model_trainer.train_advanced_model(
        X_train, y_train, X_val, y_val,
        model_name='advanced_99_model',
        epochs=150
    )
    
    # Create super ensemble for maximum accuracy
    print("\n🔄 Creating super ensemble...")
    ensemble_models = model_trainer.create_super_ensemble(
        X_train, y_train, X_val, y_val, n_models=7
    )
    
    # Final evaluation on test set
    print("\n📊 Final Evaluation on Test Set:")
    print("=" * 40)
    
    # Individual model performance
    test_metrics = model_trainer.evaluate_advanced_model(
        X_test, y_test, 'advanced_99_model'
    )
    
    # Ensemble performance
    print("\n🏆 Super Ensemble Performance:")
    ensemble_pred, ensemble_uncertainty = model_trainer.predict_super_ensemble(
        X_test, use_uncertainty=True
    )
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    ensemble_accuracy = np.mean(ensemble_pred_classes == y_test)
    
    print(f"   Ensemble Accuracy: {ensemble_accuracy:.6f}")
    
    if ensemble_accuracy >= 0.99:
        print("🎯 🏆 TARGET ACHIEVED: 99%+ ACCURACY! 🏆 🎯")
        print("🌟 ExoAI Hunter has reached world-class performance!")
    elif ensemble_accuracy >= 0.98:
        print("🔥 EXCELLENT: 98%+ accuracy achieved!")
    else:
        print(f"⚠️  Close! Need {0.99 - ensemble_accuracy:.4f} more for 99% target")
    
    # Detailed classification report
    print("\n📋 Detailed Ensemble Classification Report:")
    class_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
    print(classification_report(y_test, ensemble_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Super Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_99_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save models
    print("\n💾 Saving models...")
    model_trainer.save_model('advanced_99_model', 'models/99_accuracy/')
    
    # Save ensemble metadata
    ensemble_metadata = {
        'ensemble_accuracy': float(ensemble_accuracy),
        'individual_accuracy': float(test_metrics['accuracy']),
        'best_hyperparameters': best_params,
        'model_count': len(ensemble_models),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'target_achieved': ensemble_accuracy >= 0.99,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    import json
    with open('models/99_accuracy/ensemble_metadata.json', 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    print("✅ Training complete! Models saved to 'models/99_accuracy/'")
    
    return {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracy': test_metrics['accuracy'],
        'best_params': best_params,
        'model_trainer': model_trainer
    }

def visualize_training_progress(history):
    """
    Create comprehensive training visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    if 'f1_score' in history.history:
        axes[0, 2].plot(history.history['f1_score'], label='Training', linewidth=2)
        axes[0, 2].plot(history.history['val_f1_score'], label='Validation', linewidth=2)
        axes[0, 2].set_title('F1 Score', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'], linewidth=2)
        axes[1, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress_99_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Set up GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🖥️  Using GPU: {len(gpus)} device(s) available")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("🖥️  Using CPU for training")
    
    # Run training
    results = train_99_accuracy_model()
    
    print("\n" + "="*60)
    print("🏆 TRAINING COMPLETE!")
    print(f"🎯 Final Ensemble Accuracy: {results['ensemble_accuracy']:.6f}")
    print(f"🔬 Individual Model Accuracy: {results['individual_accuracy']:.6f}")
    print("🚀 ExoAI Hunter is ready for world-class exoplanet detection!")
    print("="*60)
