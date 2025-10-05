import axios from 'axios';

// Configure axios defaults
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout for ML processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error);
    
    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          throw new Error(data.detail || 'Invalid request data');
        case 404:
          throw new Error('Resource not found');
        case 500:
          throw new Error(data.detail || 'Internal server error');
        default:
          throw new Error(data.detail || `Request failed with status ${status}`);
      }
    } else if (error.request) {
      // Network error
      throw new Error('Network error - please check your connection');
    } else {
      // Other error
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

export const apiService = {
  // Health check
  healthCheck: () => apiClient.get('/health'),

  // Exoplanet prediction
  predictExoplanet: (data) => {
    const requestData = {
      light_curve_data: data.lightCurveData,
      metadata: data.metadata || {},
      mission: data.mission || 'kepler',
    };
    
    return apiClient.post('/predict', requestData);
  },

  // Batch prediction
  batchPredict: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return apiClient.post('/batch-predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2 minutes for batch processing
    });
  },

  // Model statistics
  getModelStats: () => apiClient.get('/models/stats'),

  // Dataset information
  getDatasetInfo: () => apiClient.get('/datasets/info'),

  // Upload and process light curve file
  uploadLightCurve: async (file, mission = 'kepler') => {
    try {
      // For CSV files, parse and extract light curve data
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        const text = await file.text();
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        
        // Find flux column (common names: flux, PDCSAP_FLUX, SAP_FLUX)
        const fluxColumnIndex = headers.findIndex(header => 
          header.toLowerCase().includes('flux') || 
          header.toLowerCase().includes('intensity')
        );
        
        if (fluxColumnIndex === -1) {
          throw new Error('No flux column found in CSV file');
        }
        
        // Extract flux data
        const lightCurveData = [];
        for (let i = 1; i < lines.length; i++) {
          if (lines[i].trim()) {
            const values = lines[i].split(',');
            const fluxValue = parseFloat(values[fluxColumnIndex]);
            if (!isNaN(fluxValue)) {
              lightCurveData.push(fluxValue);
            }
          }
        }
        
        if (lightCurveData.length < 100) {
          throw new Error('Insufficient data points. Need at least 100 flux measurements.');
        }
        
        return {
          lightCurveData,
          metadata: {
            filename: file.name,
            dataPoints: lightCurveData.length,
            uploadTime: new Date().toISOString(),
          },
          mission,
        };
      }
      
      // For JSON files
      if (file.type === 'application/json' || file.name.endsWith('.json')) {
        const text = await file.text();
        const data = JSON.parse(text);
        
        if (Array.isArray(data)) {
          return {
            lightCurveData: data,
            metadata: {
              filename: file.name,
              dataPoints: data.length,
              uploadTime: new Date().toISOString(),
            },
            mission,
          };
        } else if (data.flux || data.lightCurve) {
          return {
            lightCurveData: data.flux || data.lightCurve,
            metadata: {
              ...data.metadata,
              filename: file.name,
              uploadTime: new Date().toISOString(),
            },
            mission,
          };
        }
      }
      
      throw new Error('Unsupported file format. Please upload CSV or JSON files.');
      
    } catch (error) {
      throw new Error(`File processing failed: ${error.message}`);
    }
  },

  // Generate sample light curve data
  generateSampleData: (type = 'confirmed', mission = 'kepler') => {
    const length = 1000;
    const time = Array.from({ length }, (_, i) => i);
    let flux = [];
    
    switch (type) {
      case 'confirmed':
        // Simulate exoplanet transit with periodic dips
        flux = time.map(t => {
          const baseFlux = 1.0 + 0.01 * Math.sin(0.02 * t) + 0.005 * Math.random();
          const transitPeriod = 200;
          const transitDuration = 10;
          const transitDepth = 0.05;
          
          if (t % transitPeriod < transitDuration) {
            return baseFlux - transitDepth;
          }
          return baseFlux;
        });
        break;
        
      case 'candidate':
        // Simulate potential exoplanet with weaker signal
        flux = time.map(t => {
          const baseFlux = 1.0 + 0.02 * Math.sin(0.015 * t) + 0.01 * Math.random();
          const transitPeriod = 300;
          const transitDuration = 8;
          const transitDepth = 0.02;
          
          if (t % transitPeriod < transitDuration) {
            return baseFlux - transitDepth;
          }
          return baseFlux;
        });
        break;
        
      case 'false_positive':
        // Simulate stellar variability without transits
        flux = time.map(t => {
          return 1.0 + 0.03 * Math.sin(0.01 * t) + 0.02 * Math.sin(0.05 * t) + 0.015 * Math.random();
        });
        break;
        
      default:
        flux = time.map(() => 1.0 + 0.01 * Math.random());
    }
    
    return {
      lightCurveData: flux,
      metadata: {
        type: `sample_${type}`,
        mission,
        generated: true,
        dataPoints: length,
        generatedTime: new Date().toISOString(),
      },
      mission,
    };
  },

  // Real-time WebSocket connection for live processing updates
  connectWebSocket: (onMessage, onError) => {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws';
    
    try {
      const socket = new WebSocket(wsUrl);
      
      socket.onopen = () => {
        console.log('🔌 WebSocket connected');
      };
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
      };
      
      socket.onclose = () => {
        console.log('🔌 WebSocket disconnected');
      };
      
      return socket;
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      if (onError) onError(error);
      return null;
    }
  },

  // Utility functions
  validateLightCurveData: (data) => {
    if (!Array.isArray(data)) {
      throw new Error('Light curve data must be an array');
    }
    
    if (data.length < 100) {
      throw new Error('Light curve must contain at least 100 data points');
    }
    
    const validNumbers = data.filter(point => typeof point === 'number' && !isNaN(point));
    if (validNumbers.length < data.length * 0.9) {
      throw new Error('Light curve contains too many invalid data points');
    }
    
    return true;
  },

  formatPredictionResponse: (response) => {
    const { prediction, confidence, probability_scores, processing_time, uncertainty_estimate, explanation } = response;
    
    return {
      prediction: {
        class: prediction,
        confidence: Math.round(confidence * 100),
        processingTime: Math.round(processing_time * 1000), // Convert to ms
        uncertainty: Math.round(uncertainty_estimate * 100),
        explanation,
      },
      probabilities: {
        confirmed: Math.round((probability_scores?.CONFIRMED || 0) * 100),
        candidate: Math.round((probability_scores?.CANDIDATE || 0) * 100),
        falsePositive: Math.round((probability_scores?.FALSE_POSITIVE || 0) * 100),
      },
    };
  },
};

export default apiService;
