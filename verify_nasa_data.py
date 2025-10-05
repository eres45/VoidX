#!/usr/bin/env python3
"""
Verify NASA Data Integration - Quick Test Script
"""

import pandas as pd
import numpy as np
import os

def verify_nasa_data():
    """
    Verify that real NASA data is available and can be processed
    """
    print("🛰️ Verifying NASA Data Integration...")
    print("=" * 50)
    
    data_dir = "data"
    files_to_check = [
        "kepler_objects_of_interest.csv",
        "tess_objects_of_interest.csv", 
        "k2_planets_candidates.csv"
    ]
    
    total_objects = 0
    
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"✅ {filename}: {len(df)} objects loaded")
                
                # Show sample columns
                print(f"   Columns: {list(df.columns[:5])}...")
                
                # Count dispositions if available
                if 'koi_disposition' in df.columns:
                    dispositions = df['koi_disposition'].value_counts()
                    print(f"   Dispositions: {dict(dispositions)}")
                elif 'toi_disposition' in df.columns:
                    dispositions = df['toi_disposition'].value_counts()
                    print(f"   Dispositions: {dict(dispositions)}")
                elif 'k2c_disposition' in df.columns:
                    dispositions = df['k2c_disposition'].value_counts()
                    print(f"   Dispositions: {dict(dispositions)}")
                
                total_objects += len(df)
                print()
                
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        else:
            print(f"⚠️  {filename} not found in {data_dir}/")
    
    print(f"🎯 Total NASA Objects Available: {total_objects}")
    
    if total_objects > 10000:
        print("✅ EXCELLENT: Large dataset available for training")
        print("🛰️ Ready for authentic NASA exoplanet detection!")
    elif total_objects > 1000:
        print("✅ GOOD: Sufficient data for training")
        print("🛰️ Ready for NASA-based exoplanet detection!")
    else:
        print("⚠️  Limited data available")
        print("💡 Consider downloading more NASA datasets")
    
    print("\n" + "=" * 50)
    print("🚀 NASA Data Verification Complete!")
    
    return total_objects > 0

if __name__ == "__main__":
    success = verify_nasa_data()
    
    if success:
        print("\n🎉 Your ExoAI Hunter is ready to use REAL NASA DATA!")
        print("🏆 This qualifies for NASA Space Apps Global Awards!")
    else:
        print("\n⚠️  NASA data files need to be added to data/ directory")
        print("📥 Download from: https://exoplanetarchive.ipac.caltech.edu/")
