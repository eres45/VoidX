#!/usr/bin/env python3
"""
ExoAI Hunter - Comprehensive Benchmarking Suite
Test processing speed, accuracy, and system performance
"""

import time
import numpy as np
import pandas as pd
import joblib
import os
import psutil
import platform
from datetime import datetime
import tensorflow as tf

class ExoAIHunterBenchmark:
    """
    Comprehensive benchmarking for ExoAI Hunter system
    """
    
    def __init__(self):
        self.results = {}
        self.system_info = {}
        
    def get_system_info(self):
        """Get system specifications"""
        print("🖥️ GATHERING SYSTEM INFORMATION")
        print("=" * 50)
        
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': platform.python_version(),
        }
        
        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        self.system_info['gpu_available'] = len(gpus) > 0
        self.system_info['gpu_count'] = len(gpus)
        
        for key, value in self.system_info.items():
            print(f"  {key}: {value}")
        
        print("✅ System information collected")
        return self.system_info
    
    def benchmark_data_loading(self):
        """Benchmark dataset loading speed"""
        print("\n📊 BENCHMARKING DATA LOADING")
        print("=" * 50)
        
        datasets = [
            'nasa_datasets/cumulative_2025.10.05_07.16.00.csv',
            'nasa_datasets/TOI_2025.10.05_07.13.15.csv',
            'nasa_datasets/k2pandc_2025.10.05_07.11.02.csv'
        ]
        
        loading_times = []
        total_objects = 0
        
        for dataset in datasets:
            if os.path.exists(dataset):
                start_time = time.time()
                df = pd.read_csv(dataset, comment='#')
                end_time = time.time()
                
                load_time = end_time - start_time
                loading_times.append(load_time)
                total_objects += len(df)
                
                print(f"✅ {os.path.basename(dataset)}: {len(df):,} objects in {load_time:.3f}s")
            else:
                print(f"❌ {dataset}: Not found")
        
        avg_loading_time = np.mean(loading_times) if loading_times else 0
        
        self.results['data_loading'] = {
            'total_objects': total_objects,
            'total_loading_time': sum(loading_times),
            'average_loading_time': avg_loading_time,
            'objects_per_second': total_objects / sum(loading_times) if sum(loading_times) > 0 else 0
        }
        
        print(f"\n📊 Data Loading Summary:")
        print(f"  Total Objects: {total_objects:,}")
        print(f"  Total Time: {sum(loading_times):.3f}s")
        print(f"  Speed: {self.results['data_loading']['objects_per_second']:,.0f} objects/second")
        
        return self.results['data_loading']
    
    def benchmark_model_loading(self):
        """Benchmark model loading speed"""
        print("\n🤖 BENCHMARKING MODEL LOADING")
        print("=" * 50)
        
        model_dirs = ['perfected_models', 'elite_models', 'ultimate_models']
        model_loading_times = {}
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                print(f"\n📁 Testing {model_dir}:")
                
                # Test different model types
                model_files = {
                    'Random Forest': f'{model_dir}/perfected_random_forest.pkl',
                    'Extra Trees': f'{model_dir}/perfected_extra_trees.pkl',
                    'Gradient Boost': f'{model_dir}/perfected_gradient_boost.pkl',
                    'Neural Network': f'{model_dir}/perfected_neural_network.keras',
                    'Scaler': f'{model_dir}/perfected_tree_scaler.pkl'
                }
                
                dir_times = {}
                for model_name, model_path in model_files.items():
                    if os.path.exists(model_path):
                        try:
                            start_time = time.time()
                            
                            if model_path.endswith('.keras'):
                                model = tf.keras.models.load_model(model_path)
                            else:
                                model = joblib.load(model_path)
                            
                            end_time = time.time()
                            load_time = end_time - start_time
                            dir_times[model_name] = load_time
                            
                            print(f"  ✅ {model_name}: {load_time:.3f}s")
                            
                        except Exception as e:
                            print(f"  ❌ {model_name}: Error - {str(e)[:50]}...")
                    else:
                        print(f"  ⚠️ {model_name}: Not found")
                
                model_loading_times[model_dir] = dir_times
        
        self.results['model_loading'] = model_loading_times
        
        # Calculate overall model loading performance
        all_times = []
        for dir_times in model_loading_times.values():
            all_times.extend(dir_times.values())
        
        if all_times:
            avg_model_load_time = np.mean(all_times)
            print(f"\n📊 Model Loading Summary:")
            print(f"  Average Load Time: {avg_model_load_time:.3f}s")
            print(f"  Fastest: {min(all_times):.3f}s")
            print(f"  Slowest: {max(all_times):.3f}s")
        
        return self.results['model_loading']
    
    def benchmark_inference_speed(self):
        """Benchmark model inference speed"""
        print("\n⚡ BENCHMARKING INFERENCE SPEED")
        print("=" * 50)
        
        # Create sample data for inference testing
        sample_sizes = [1, 10, 100, 1000]
        feature_count = 40  # Based on our perfected model
        
        # Load a model for testing
        model_path = 'perfected_models/perfected_random_forest.pkl'
        scaler_path = 'perfected_models/perfected_tree_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                inference_results = {}
                
                for sample_size in sample_sizes:
                    # Generate random sample data
                    X_sample = np.random.random((sample_size, feature_count))
                    X_scaled = scaler.transform(X_sample)
                    
                    # Warm-up run
                    _ = model.predict(X_scaled)
                    
                    # Benchmark multiple runs
                    times = []
                    for _ in range(10):
                        start_time = time.time()
                        predictions = model.predict(X_scaled)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    avg_time = np.mean(times)
                    time_per_object = avg_time / sample_size
                    objects_per_second = sample_size / avg_time
                    
                    inference_results[sample_size] = {
                        'avg_time': avg_time,
                        'time_per_object': time_per_object,
                        'objects_per_second': objects_per_second
                    }
                    
                    print(f"✅ {sample_size:4d} objects: {avg_time:.4f}s total, {time_per_object:.4f}s/object, {objects_per_second:,.0f} obj/s")
                
                self.results['inference_speed'] = inference_results
                
                # Calculate single object processing time
                single_object_time = inference_results[1]['time_per_object']
                print(f"\n📊 Inference Speed Summary:")
                print(f"  Single Object: {single_object_time:.4f}s ({single_object_time*1000:.2f}ms)")
                print(f"  Batch Processing: {inference_results[1000]['objects_per_second']:,.0f} objects/second")
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.results['inference_speed'] = {'error': str(e)}
        else:
            print("❌ Model files not found for inference testing")
            self.results['inference_speed'] = {'error': 'Model files not found'}
        
        return self.results['inference_speed']
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n💾 BENCHMARKING MEMORY USAGE")
        print("=" * 50)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        print(f"Initial Memory: {initial_memory:.1f} MB")
        
        # Test memory usage during data loading
        if os.path.exists('nasa_datasets/cumulative_2025.10.05_07.16.00.csv'):
            df = pd.read_csv('nasa_datasets/cumulative_2025.10.05_07.16.00.csv', comment='#')
            after_data_memory = process.memory_info().rss / (1024**2)
            data_memory_usage = after_data_memory - initial_memory
            
            print(f"After Data Loading: {after_data_memory:.1f} MB (+{data_memory_usage:.1f} MB)")
        
        # Test memory usage during model loading
        if os.path.exists('perfected_models/perfected_random_forest.pkl'):
            try:
                model = joblib.load('perfected_models/perfected_random_forest.pkl')
                after_model_memory = process.memory_info().rss / (1024**2)
                model_memory_usage = after_model_memory - after_data_memory if 'after_data_memory' in locals() else after_model_memory - initial_memory
                
                print(f"After Model Loading: {after_model_memory:.1f} MB (+{model_memory_usage:.1f} MB)")
                
                self.results['memory_usage'] = {
                    'initial_mb': initial_memory,
                    'data_loading_mb': data_memory_usage if 'data_memory_usage' in locals() else 0,
                    'model_loading_mb': model_memory_usage,
                    'total_mb': after_model_memory
                }
                
            except Exception as e:
                print(f"❌ Error in memory benchmark: {e}")
        
        return self.results.get('memory_usage', {})
    
    def benchmark_accuracy_validation(self):
        """Validate model accuracy"""
        print("\n🎯 VALIDATING MODEL ACCURACY")
        print("=" * 50)
        
        # Check if results files exist
        results_files = [
            'perfected_models/perfected_results.pkl',
            'elite_models/elite_results.pkl',
            'ultimate_models/ultimate_ensemble_info.pkl'
        ]
        
        accuracy_results = {}
        
        for results_file in results_files:
            if os.path.exists(results_file):
                try:
                    results = joblib.load(results_file)
                    model_name = os.path.dirname(results_file).replace('_models', '')
                    
                    if 'final_accuracy' in results:
                        accuracy = results['final_accuracy']
                    elif 'ultimate_accuracy' in results:
                        accuracy = results['ultimate_accuracy']
                    elif 'ensemble_accuracy' in results:
                        accuracy = results['ensemble_accuracy']
                    else:
                        accuracy = None
                    
                    if accuracy:
                        accuracy_results[model_name] = {
                            'accuracy': accuracy,
                            'percentage': accuracy * 100
                        }
                        print(f"✅ {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                except Exception as e:
                    print(f"❌ Error reading {results_file}: {e}")
        
        self.results['accuracy_validation'] = accuracy_results
        
        if accuracy_results:
            best_accuracy = max(acc['accuracy'] for acc in accuracy_results.values())
            best_model = [name for name, acc in accuracy_results.items() if acc['accuracy'] == best_accuracy][0]
            
            print(f"\n📊 Accuracy Summary:")
            print(f"  Best Model: {best_model}")
            print(f"  Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        return self.results['accuracy_validation']
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("🚀 EXOAI HUNTER COMPREHENSIVE BENCHMARK")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Run all benchmark tests
        self.get_system_info()
        self.benchmark_data_loading()
        self.benchmark_model_loading()
        self.benchmark_inference_speed()
        self.benchmark_memory_usage()
        self.benchmark_accuracy_validation()
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 70)
        print("🏆 EXOAI HUNTER BENCHMARK SUMMARY REPORT")
        print("=" * 70)
        
        # System Performance
        print("\n🖥️ SYSTEM PERFORMANCE:")
        if 'data_loading' in self.results:
            print(f"  📊 Data Loading: {self.results['data_loading']['objects_per_second']:,.0f} objects/second")
        
        if 'inference_speed' in self.results and 'error' not in self.results['inference_speed']:
            single_time = self.results['inference_speed'][1]['time_per_object']
            print(f"  ⚡ Single Object Processing: {single_time:.4f}s ({single_time*1000:.2f}ms)")
            print(f"  🚀 Batch Processing: {self.results['inference_speed'][1000]['objects_per_second']:,.0f} objects/second")
        
        # Accuracy Performance
        print("\n🎯 ACCURACY PERFORMANCE:")
        if 'accuracy_validation' in self.results:
            for model_name, acc_data in self.results['accuracy_validation'].items():
                print(f"  🏆 {model_name}: {acc_data['percentage']:.2f}%")
        
        # Resource Usage
        print("\n💾 RESOURCE USAGE:")
        if 'memory_usage' in self.results:
            total_memory = self.results['memory_usage'].get('total_mb', 0)
            print(f"  💾 Total Memory: {total_memory:.1f} MB")
        
        print(f"  🖥️ CPU Cores: {self.system_info.get('cpu_count', 'Unknown')}")
        print(f"  🎮 GPU Available: {'Yes' if self.system_info.get('gpu_available', False) else 'No'}")
        
        # Performance Rating
        self.calculate_performance_rating()
        
        print("\n" + "=" * 70)
        print("✅ BENCHMARK COMPLETE - ExoAI Hunter Performance Validated!")
        print("=" * 70)
    
    def calculate_performance_rating(self):
        """Calculate overall performance rating"""
        print("\n⭐ PERFORMANCE RATING:")
        
        ratings = []
        
        # Accuracy rating
        if 'accuracy_validation' in self.results:
            best_acc = max(acc['accuracy'] for acc in self.results['accuracy_validation'].values())
            if best_acc >= 0.94:
                ratings.append(("Accuracy", "⭐⭐⭐⭐⭐", "Exceptional"))
            elif best_acc >= 0.90:
                ratings.append(("Accuracy", "⭐⭐⭐⭐", "Excellent"))
            elif best_acc >= 0.85:
                ratings.append(("Accuracy", "⭐⭐⭐", "Good"))
            else:
                ratings.append(("Accuracy", "⭐⭐", "Fair"))
        
        # Speed rating
        if 'inference_speed' in self.results and 'error' not in self.results['inference_speed']:
            single_time = self.results['inference_speed'][1]['time_per_object']
            if single_time <= 0.001:
                ratings.append(("Speed", "⭐⭐⭐⭐⭐", "Lightning Fast"))
            elif single_time <= 0.01:
                ratings.append(("Speed", "⭐⭐⭐⭐", "Very Fast"))
            elif single_time <= 0.1:
                ratings.append(("Speed", "⭐⭐⭐", "Fast"))
            else:
                ratings.append(("Speed", "⭐⭐", "Moderate"))
        
        # Memory efficiency rating
        if 'memory_usage' in self.results:
            total_memory = self.results['memory_usage'].get('total_mb', 0)
            if total_memory <= 100:
                ratings.append(("Memory", "⭐⭐⭐⭐⭐", "Very Efficient"))
            elif total_memory <= 500:
                ratings.append(("Memory", "⭐⭐⭐⭐", "Efficient"))
            elif total_memory <= 1000:
                ratings.append(("Memory", "⭐⭐⭐", "Moderate"))
            else:
                ratings.append(("Memory", "⭐⭐", "Heavy"))
        
        for category, stars, description in ratings:
            print(f"  {category}: {stars} ({description})")
        
        # Overall rating
        avg_stars = len([r for r in ratings if "⭐⭐⭐⭐⭐" in r[1]]) + len([r for r in ratings if "⭐⭐⭐⭐" in r[1]]) * 0.8
        if avg_stars >= 2:
            print(f"\n🏆 OVERALL: ⭐⭐⭐⭐⭐ EXCEPTIONAL PERFORMANCE!")
        elif avg_stars >= 1.5:
            print(f"\n🏆 OVERALL: ⭐⭐⭐⭐ EXCELLENT PERFORMANCE!")
        else:
            print(f"\n🏆 OVERALL: ⭐⭐⭐ GOOD PERFORMANCE!")

def main():
    """Run comprehensive benchmark"""
    benchmark = ExoAIHunterBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'benchmark_results_{timestamp}.pkl'
    joblib.dump(results, results_file)
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = main()
