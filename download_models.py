#!/usr/bin/env python3
"""
Download pre-trained ExoAI Hunter models from Google Drive

This script downloads the pre-trained models for ExoAI Hunter
since they are too large to store in GitHub repository.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Google Drive file IDs - ExoAI Hunter Pre-trained Models
GOOGLE_DRIVE_FILES = {
    # Google Drive folder: https://drive.google.com/drive/folders/1SLu-iH4g7YL20a8sJL9kyFgG6F4Pg01T?usp=sharing
    "models_folder_id": "1SLu-iH4g7YL20a8sJL9kyFgG6F4Pg01T"
}

# Model file information (Total: 1,039.9 MB)
MODEL_INFO = {
    "perfected_stacking.pkl": {"size_mb": 692.9, "description": "Main stacking ensemble model"},
    "perfected_gradient_boost.pkl": {"size_mb": 165.7, "description": "Gradient Boosting classifier"},
    "perfected_extra_trees.pkl": {"size_mb": 129.4, "description": "Extra Trees classifier"},
    "perfected_random_forest.pkl": {"size_mb": 51.3, "description": "Random Forest classifier"},
    "perfected_neural_network.keras": {"size_mb": 0.7, "description": "Neural Network model"},
    "perfected_nn_scaler.pkl": {"size_mb": 0.002, "description": "Neural Network scaler"},
    "perfected_tree_scaler.pkl": {"size_mb": 0.001, "description": "Tree models scaler"},
    "perfected_results.pkl": {"size_mb": 0.0002, "description": "Training results metadata"}
}

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large file download confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_models():
    """Download pre-trained models for ExoAI Hunter"""
    
    print("🤖 ExoAI Hunter Model Downloader")
    print("=" * 70)
    print("🎯 NASA Space Apps Challenge 2025 - Team VoidX")
    print("🚀 ExoAI Hunter: 94.51% Accuracy on Real NASA Data")
    print("=" * 70)
    
    # Display model information
    total_size = sum(info["size_mb"] for info in MODEL_INFO.values())
    print(f"📊 TRAINED MODELS INFORMATION:")
    print(f"   📦 Total Size: {total_size:.1f} MB (1.04 GB)")
    print(f"   🤖 Model Count: {len(MODEL_INFO)} files")
    print(f"   🛰️ Training Data: 21,271 NASA objects (Kepler, K2, TESS)")
    print(f"   🎯 Accuracy: 94.51% on authentic NASA exoplanet data")
    print(f"   ⚡ Processing: 52.6ms ultra-fast inference")
    
    print(f"\n❓ WHY NOT IN GITHUB?")
    print(f"   🚫 GitHub Limits: 100MB per file, 1GB per repository")
    print(f"   📈 Our Models: {total_size:.1f}MB total (exceeds limits)")
    print(f"   🔧 Solution: Google Drive hosting with automated download")
    
    print(f"\n📁 MODEL BREAKDOWN:")
    for filename, info in MODEL_INFO.items():
        print(f"   📄 {filename:<30} {info['size_mb']:>8.1f}MB - {info['description']}")
    
    # Create models directory
    models_dir = Path("perfected_models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Models directory: perfected_models/")
    print(f"🔗 Google Drive: https://drive.google.com/drive/folders/1SLu-iH4g7YL20a8sJL9kyFgG6F4Pg01T")
    
    # Check if models already exist
    model_files = [
        "perfected_extra_trees.pkl",
        "perfected_gradient_boost.pkl", 
        "perfected_neural_network.keras",
        "perfected_nn_scaler.pkl",
        "perfected_random_forest.pkl",
        "perfected_results.pkl",
        "perfected_stacking.pkl",
        "perfected_tree_scaler.pkl"
    ]
    
    existing_models = [f for f in model_files if (models_dir / f).exists() and (models_dir / f).stat().st_size > 1000]
    
    if len(existing_models) == len(model_files):
        print("✅ All models already downloaded!")
        print("🚀 You can now run: python start_exoai_hunter.py")
        return
    
    print(f"📥 Need to download {len(model_files) - len(existing_models)} model files...")
    
    # Download from Google Drive
    if GOOGLE_DRIVE_FILES["models_package"] == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        print("⚠️  Google Drive file ID not configured!")
        print("📝 Please update the GOOGLE_DRIVE_FILES dictionary with actual file IDs")
        print("📋 Instructions:")
        print("   1. Upload models to Google Drive")
        print("   2. Get shareable link")
        print("   3. Extract file ID from link")
        print("   4. Update download_models.py")
        
        # Create placeholder files for now
        print("\n🔄 Creating placeholder files for development...")
        for model_file in model_files:
            model_path = models_dir / model_file
            if not model_path.exists() or model_path.stat().st_size < 1000:
                with open(model_path, 'w') as f:
                    f.write(f"# Placeholder for {model_file}\n")
                    f.write("# Run 'python download_models.py' after configuring Google Drive links\n")
                print(f"   📄 Created placeholder: {model_file}")
        
        print("\n⚠️  Note: Placeholders created. Update Google Drive IDs for full functionality.")
        return
    
    try:
        print("📥 Downloading models from Google Drive...")
        zip_path = "models.zip"
        
        download_file_from_google_drive(GOOGLE_DRIVE_FILES["models_package"], zip_path)
        
        print("📦 Extracting models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up zip file
        os.remove(zip_path)
        
        print("✅ Models downloaded successfully!")
        print("🚀 You can now run: python start_exoai_hunter.py")
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print("📝 Please check the Google Drive file ID and try again")

if __name__ == "__main__":
    download_models()
