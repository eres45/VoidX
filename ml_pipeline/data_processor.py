"""
ExoAI Hunter - Data Processing Pipeline
Advanced preprocessing for NASA exoplanet datasets (Kepler, K2, TESS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import astropy.units as u
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDataProcessor:
    """
    Advanced data preprocessing pipeline for exoplanet detection
    Handles Kepler, K2, and TESS mission data with specialized techniques
    """
    
    def __init__(self, mission_type='kepler'):
        self.mission_type = mission_type.lower()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Mission-specific parameters
        self.mission_configs = {
            'kepler': {
                'cadence': 29.4 * u.minute,  # Long cadence
                'noise_threshold': 0.1,
                'detrend_window': 101
            },
            'k2': {
                'cadence': 29.4 * u.minute,
                'noise_threshold': 0.15,
                'detrend_window': 81
            },
            'tess': {
                'cadence': 2 * u.minute,  # Short cadence
                'noise_threshold': 0.05,
                'detrend_window': 201
            }
        }
    
    def load_nasa_dataset(self, file_path):
        """
        Load NASA exoplanet dataset with mission-specific handling
        """
        try:
            df = pd.read_csv(file_path, comment='#', low_memory=False)
            
            # Mission-specific column mapping
            if self.mission_type == 'kepler':
                # KOI dataset columns
                target_col = 'koi_disposition'
                period_col = 'koi_period'
                depth_col = 'koi_depth'
            elif self.mission_type == 'k2':
                # K2 dataset columns  
                target_col = 'k2c_disp'  
                period_col = 'pl_orbper'
                depth_col = 'pl_trandep'
            else:  # TESS
                # TOI dataset columns
                target_col = 'tfopwg_disp'
                period_col = 'pl_orbper' 
                depth_col = 'pl_trandep'
            
            print(f"📊 Loaded {len(df)} objects from {self.mission_type.upper()} dataset")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def preprocess_light_curve(self, flux_data, time_data=None):
        """
        Advanced light curve preprocessing with detrending and normalization
        """
        flux = np.array(flux_data, dtype=float)
        
        # Remove NaN values
        valid_mask = np.isfinite(flux)
        flux = flux[valid_mask]
        
        if len(flux) < 100:
            raise ValueError("Insufficient data points for processing")
        
        # Step 1: Outlier removal using sigma clipping
        flux = self._remove_outliers(flux, sigma=3.0)
        
        # Step 2: Detrending to remove stellar variability
        flux_detrended = self._detrend_light_curve(flux)
        
        # Step 3: Normalization
        flux_normalized = self._normalize_flux(flux_detrended)
        
        # Step 4: Gap filling for missing data
        flux_processed = self._fill_gaps(flux_normalized)
        
        return flux_processed
    
    def _remove_outliers(self, flux, sigma=3.0):
        """Remove outliers using sigma clipping"""
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        threshold = sigma * 1.4826 * mad  # Convert MAD to std
        
        mask = np.abs(flux - median_flux) < threshold
        return flux[mask]
    
    def _detrend_light_curve(self, flux):
        """
        Detrend light curve using Savitzky-Golay filter
        Removes long-term stellar variability while preserving transits
        """
        config = self.mission_configs[self.mission_type]
        window_length = min(config['detrend_window'], len(flux)//4)
        
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length < 5:
            window_length = 5
        
        # Apply Savitzky-Golay filter for detrending
        trend = signal.savgol_filter(flux, window_length, polyorder=2)
        detrended = flux / trend
        
        return detrended
    
    def _normalize_flux(self, flux):
        """Normalize flux to have zero mean and unit variance"""
        return (flux - np.mean(flux)) / np.std(flux)
    
    def _fill_gaps(self, flux):
        """Fill gaps in light curve data using interpolation"""
        if len(flux) < 1000:
            # For short light curves, use linear interpolation
            x = np.arange(len(flux))
            valid_mask = np.isfinite(flux)
            flux_filled = np.interp(x, x[valid_mask], flux[valid_mask])
        else:
            # For longer light curves, use median filter
            flux_filled = median_filter(flux, size=3)
        
        return flux_filled
    
    def extract_features(self, flux_data, period=None):
        """
        Extract comprehensive features for ML model
        Combines statistical, spectral, and transit-specific features
        """
        flux = np.array(flux_data)
        features = {}
        
        # Statistical features
        features.update(self._extract_statistical_features(flux))
        
        # Spectral features 
        features.update(self._extract_spectral_features(flux))
        
        # Transit-specific features
        if period is not None:
            features.update(self._extract_transit_features(flux, period))
        
        # Shape-based features
        features.update(self._extract_shape_features(flux))
        
        return features
    
    def _extract_statistical_features(self, flux):
        """Extract statistical features from light curve"""
        return {
            'mean': np.mean(flux),
            'std': np.std(flux),
            'variance': np.var(flux),
            'skewness': float(pd.Series(flux).skew()),
            'kurtosis': float(pd.Series(flux).kurtosis()),
            'median': np.median(flux),
            'mad': np.median(np.abs(flux - np.median(flux))),
            'percentile_25': np.percentile(flux, 25),
            'percentile_75': np.percentile(flux, 75),
            'range': np.max(flux) - np.min(flux),
            'iqr': np.percentile(flux, 75) - np.percentile(flux, 25)
        }
    
    def _extract_spectral_features(self, flux):
        """Extract frequency domain features"""
        # Compute power spectral density
        freqs, psd = signal.periodogram(flux)
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(psd, height=np.percentile(psd, 90))[0]
        
        return {
            'dominant_frequency': freqs[np.argmax(psd)] if len(freqs) > 0 else 0,
            'spectral_centroid': np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0,
            'spectral_bandwidth': np.sqrt(np.sum(((freqs - np.sum(freqs * psd) / np.sum(psd))**2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0,
            'num_peaks': len(peak_indices),
            'max_power': np.max(psd),
            'total_power': np.sum(psd)
        }
    
    def _extract_transit_features(self, flux, period):
        """Extract transit-specific features"""
        try:
            # Phase fold the light curve
            phase = np.arange(len(flux)) % int(period * 48)  # Assuming ~30min cadence
            phase_normalized = phase / np.max(phase)
            
            # Find transit depth and duration
            transit_mask = (phase_normalized > 0.45) & (phase_normalized < 0.55)
            
            if np.sum(transit_mask) > 0:
                transit_depth = np.median(flux[~transit_mask]) - np.median(flux[transit_mask])
                transit_duration = np.sum(transit_mask) / len(flux)
            else:
                transit_depth = 0
                transit_duration = 0
                
            return {
                'transit_depth': transit_depth,
                'transit_duration': transit_duration,
                'period': period,
                'depth_to_duration_ratio': transit_depth / transit_duration if transit_duration > 0 else 0
            }
        except:
            return {
                'transit_depth': 0,
                'transit_duration': 0, 
                'period': period,
                'depth_to_duration_ratio': 0
            }
    
    def _extract_shape_features(self, flux):
        """Extract light curve shape features"""
        # Detect dips and peaks
        dip_indices = signal.find_peaks(-flux, prominence=0.1)[0]
        peak_indices = signal.find_peaks(flux, prominence=0.1)[0]
        
        return {
            'num_dips': len(dip_indices),
            'num_peaks': len(peak_indices),
            'deepest_dip': np.min(flux) if len(flux) > 0 else 0,
            'highest_peak': np.max(flux) if len(flux) > 0 else 0,
            'asymmetry': self._calculate_asymmetry(flux)
        }
    
    def _calculate_asymmetry(self, flux):
        """Calculate asymmetry measure of light curve"""
        median_flux = np.median(flux)
        left_side = flux[flux < median_flux]
        right_side = flux[flux > median_flux]
        
        if len(left_side) > 0 and len(right_side) > 0:
            return (np.mean(right_side) - median_flux) / (median_flux - np.mean(left_side))
        return 0
    
    def prepare_training_data(self, dataset, feature_columns, target_column):
        """
        Prepare data for machine learning training
        """
        # Extract features and labels
        X = dataset[feature_columns].copy()
        y = dataset[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Encode target labels
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.feature_names = feature_columns
        
        print(f"✅ Training data prepared:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Classes: {len(np.unique(y))}")
        
        return X_train, X_val, y_train, y_val
    
    def visualize_light_curve(self, flux_data, title="Light Curve", save_path=None):
        """
        Create publication-quality light curve visualization
        """
        plt.figure(figsize=(12, 6))
        
        time_points = np.arange(len(flux_data))
        plt.plot(time_points, flux_data, 'b-', alpha=0.7, linewidth=0.8)
        
        plt.xlabel('Time (data points)')
        plt.ylabel('Normalized Flux')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Highlight potential transit events
        flux_array = np.array(flux_data)
        threshold = np.median(flux_array) - 2 * np.std(flux_array)
        transit_mask = flux_array < threshold
        
        if np.any(transit_mask):
            plt.fill_between(time_points, flux_array, threshold, 
                           where=transit_mask, alpha=0.3, color='red',
                           label='Potential Transit Events')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_processing_report(self, original_data, processed_data):
        """
        Generate comprehensive data processing report
        """
        report = {
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'mission_type': self.mission_type,
            'original_data_points': len(original_data),
            'processed_data_points': len(processed_data),
            'data_retention_rate': len(processed_data) / len(original_data),
            'noise_reduction': np.std(original_data) / np.std(processed_data),
            'signal_quality_score': self._calculate_quality_score(processed_data)
        }
        
        return report
    
    def _calculate_quality_score(self, flux_data):
        """Calculate signal quality score (0-1 scale)"""
        # Based on signal-to-noise ratio and data completeness
        snr = np.mean(flux_data) / np.std(flux_data) if np.std(flux_data) > 0 else 0
        completeness = 1 - (np.sum(np.isnan(flux_data)) / len(flux_data))
        
        # Normalize and combine metrics
        snr_normalized = min(abs(snr) / 10, 1.0)  # Normalize SNR to 0-1
        quality_score = (snr_normalized + completeness) / 2
        
        return min(quality_score, 1.0)
