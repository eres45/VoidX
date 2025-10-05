#!/usr/bin/env python3
"""
Create a zip package of ExoAI Hunter models for Google Drive upload

This script creates a zip file containing all the trained models
that can be uploaded to Google Drive for sharing.
"""

import os
import zipfile
from pathlib import Path

def create_models_package():
    """Create a zip package of all trained models"""
    
    print("📦 ExoAI Hunter Models Packager")
    print("=" * 50)
    
    models_dir = Path("perfected_models")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print("   Please ensure 'perfected_models/' directory exists")
        return
    
    # List of model files to package
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
    
    # Check which files exist
    existing_files = []
    total_size = 0
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists() and model_path.stat().st_size > 1000:  # Skip placeholder files
            existing_files.append(model_file)
            size = model_path.stat().st_size
            total_size += size
            print(f"✅ Found: {model_file} ({size/1024/1024:.1f} MB)")
        else:
            print(f"❌ Missing: {model_file}")
    
    if not existing_files:
        print("❌ No valid model files found!")
        return
    
    print(f"\n📊 Total size: {total_size/1024/1024:.1f} MB")
    print(f"📦 Files to package: {len(existing_files)}")
    
    # Create zip file
    zip_filename = "exoai_hunter_models.zip"
    
    print(f"\n🔄 Creating {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for model_file in existing_files:
            model_path = models_dir / model_file
            zipf.write(model_path, f"perfected_models/{model_file}")
            print(f"   📄 Added: {model_file}")
    
    # Check final zip size
    zip_size = os.path.getsize(zip_filename)
    print(f"\n✅ Package created: {zip_filename}")
    print(f"📊 Compressed size: {zip_size/1024/1024:.1f} MB")
    print(f"🗜️  Compression ratio: {(1 - zip_size/total_size)*100:.1f}%")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Upload {zip_filename} to Google Drive")
    print(f"   2. Set sharing to 'Anyone with the link can view'")
    print(f"   3. Copy the file ID from the shareable link")
    print(f"   4. Update download_models.py with the file ID")
    print(f"   5. Commit and push the updated download script")
    
    print(f"\n🌟 Ready for Google Drive upload!")

if __name__ == "__main__":
    create_models_package()
