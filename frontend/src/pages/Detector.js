import React, { useState, useCallback, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  LinearProgress,
  Alert,
  Chip,
  Paper,
  Tabs,
  Tab,
  Divider,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Science as ScienceIcon,
  TrendingUp as TrendingUpIcon,
  Visibility as VisibilityIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import Plot from 'react-plotly.js';
import { apiService } from '../services/apiService';

const Detector = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [lightCurveData, setLightCurveData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedMission, setSelectedMission] = useState('kepler');
  const [processingTime, setProcessingTime] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  // Dropzone configuration
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setUploadedFile(file);

    try {
      const processedData = await apiService.uploadLightCurve(file, selectedMission);
      setLightCurveData(processedData);
      console.log('✅ File processed successfully:', processedData);
    } catch (err) {
      setError(err.message);
      console.error('❌ File processing error:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedMission]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  // Generate sample data
  const generateSampleData = (type) => {
    setLoading(true);
    setError(null);

    try {
      const sampleData = apiService.generateSampleData(type, selectedMission);
      setLightCurveData(sampleData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Run AI prediction
  const runPrediction = async () => {
    if (!lightCurveData) {
      setError('Please upload or generate light curve data first');
      return;
    }

    setLoading(true);
    setError(null);
    const startTime = Date.now();

    try {
      const response = await apiService.predictExoplanet({
        lightCurveData: lightCurveData.lightCurveData,
        metadata: lightCurveData.metadata,
        mission: selectedMission,
      });

      const formattedPrediction = apiService.formatPredictionResponse(response.data);
      setPrediction(formattedPrediction);
      setProcessingTime(Date.now() - startTime);

      console.log('🎯 Prediction completed:', formattedPrediction);
    } catch (err) {
      setError(err.message);
      console.error('❌ Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Create light curve plot
  const createLightCurvePlot = () => {
    if (!lightCurveData) return null;

    const data = lightCurveData.lightCurveData;
    const timePoints = Array.from({ length: data.length }, (_, i) => i);

    return {
      data: [
        {
          x: timePoints,
          y: data,
          type: 'scatter',
          mode: 'lines',
          name: 'Flux',
          line: {
            color: '#00a8cc',
            width: 1,
          },
        },
      ],
      layout: {
        title: {
          text: `Light Curve - ${selectedMission.toUpperCase()} Mission`,
          font: { color: '#ffffff', size: 16 },
        },
        xaxis: {
          title: 'Time (data points)',
          color: '#b0b0b0',
          gridcolor: 'rgba(255, 255, 255, 0.1)',
        },
        yaxis: {
          title: 'Normalized Flux',
          color: '#b0b0b0',
          gridcolor: 'rgba(255, 255, 255, 0.1)',
        },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        font: { color: '#ffffff' },
        showlegend: false,
      },
      config: {
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
        displaylogo: false,
      },
    };
  };

  // Mission selector
  const missionOptions = [
    { value: 'kepler', label: 'Kepler', color: '#00a8cc' },
    { value: 'k2', label: 'K2', color: '#ff6b35' },
    { value: 'tess', label: 'TESS', color: '#00ff88' },
  ];

  // Sample data types
  const sampleTypes = [
    { type: 'confirmed', label: 'Confirmed Exoplanet', color: 'success' },
    { type: 'candidate', label: 'Planet Candidate', color: 'warning' },
    { type: 'false_positive', label: 'False Positive', color: 'error' },
  ];

  // Get prediction color and icon
  const getPredictionStyle = (predictionClass) => {
    switch (predictionClass) {
      case 'CONFIRMED':
        return { color: '#00ff88', icon: '🌍', label: 'Confirmed Exoplanet' };
      case 'CANDIDATE':
        return { color: '#ffaa00', icon: '🔍', label: 'Planet Candidate' };
      case 'FALSE_POSITIVE':
        return { color: '#ff4444', icon: '❌', label: 'False Positive' };
      default:
        return { color: '#b0b0b0', icon: '❓', label: 'Unknown' };
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          🔬 ExoAI Detector
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Real-time AI-powered exoplanet detection from light curve data
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Left Panel - Data Input */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: 'fit-content', mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📡 Data Input
              </Typography>
              
              {/* Mission Selector */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Select Mission:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {missionOptions.map((mission) => (
                    <Chip
                      key={mission.value}
                      label={mission.label}
                      onClick={() => setSelectedMission(mission.value)}
                      variant={selectedMission === mission.value ? 'filled' : 'outlined'}
                      sx={{
                        backgroundColor: selectedMission === mission.value ? mission.color : 'transparent',
                        borderColor: mission.color,
                        color: selectedMission === mission.value ? '#000' : mission.color,
                      }}
                    />
                  ))}
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* File Upload */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Upload Light Curve Data:
                </Typography>
                <Paper
                  {...getRootProps()}
                  sx={{
                    p: 3,
                    textAlign: 'center',
                    border: '2px dashed',
                    borderColor: isDragActive ? 'primary.main' : 'grey.600',
                    backgroundColor: isDragActive ? 'action.hover' : 'transparent',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderColor: 'primary.main',
                      backgroundColor: 'action.hover',
                    },
                  }}
                >
                  <input {...getInputProps()} />
                  <UploadIcon sx={{ fontSize: 48, mb: 2, color: 'primary.main' }} />
                  <Typography variant="body1" gutterBottom>
                    {isDragActive ? 'Drop files here...' : 'Drag & drop or click to select'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Supports CSV and JSON files (max 50MB)
                  </Typography>
                </Paper>
                {uploadedFile && (
                  <Box sx={{ mt: 1 }}>
                    <Chip
                      label={uploadedFile.name}
                      size="small"
                      onDelete={() => {
                        setUploadedFile(null);
                        setLightCurveData(null);
                      }}
                    />
                  </Box>
                )}
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Sample Data */}
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Or use sample data:
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {sampleTypes.map((sample) => (
                    <Button
                      key={sample.type}
                      variant="outlined"
                      size="small"
                      onClick={() => generateSampleData(sample.type)}
                      disabled={loading}
                      sx={{ justifyContent: 'flex-start' }}
                    >
                      {sample.label}
                    </Button>
                  ))}
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* Analysis Button */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🚀 Analysis
              </Typography>
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={<PlayIcon />}
                onClick={runPrediction}
                disabled={!lightCurveData || loading}
                sx={{ mb: 2 }}
              >
                {loading ? 'Processing...' : 'Analyze Exoplanet'}
              </Button>
              
              {processingTime && (
                <Typography variant="caption" color="text.secondary" align="center" display="block">
                  ⚡ Processed in {processingTime}ms
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Right Panel - Visualization and Results */}
        <Grid item xs={12} lg={8}>
          {/* Loading Bar */}
          {loading && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <ScienceIcon sx={{ mr: 1 }} />
                  <Typography variant="body1">
                    AI processing in progress...
                  </Typography>
                </Box>
                <LinearProgress />
              </CardContent>
            </Card>
          )}

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {/* Results */}
          <AnimatePresence>
            {prediction && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
              >
                <Card sx={{ mb: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      🎯 AI Prediction Results
                    </Typography>
                    
                    <Grid container spacing={3}>
                      {/* Main Prediction */}
                      <Grid item xs={12} md={6}>
                        <Box
                          sx={{
                            p: 3,
                            textAlign: 'center',
                            border: 2,
                            borderColor: getPredictionStyle(prediction.prediction.class).color,
                            borderRadius: 2,
                          }}
                        >
                          <Typography variant="h3" sx={{ mb: 1 }}>
                            {getPredictionStyle(prediction.prediction.class).icon}
                          </Typography>
                          <Typography variant="h5" gutterBottom>
                            {getPredictionStyle(prediction.prediction.class).label}
                          </Typography>
                          <Typography variant="h4" fontWeight="bold" sx={{ mb: 1 }}>
                            {prediction.prediction.confidence}%
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Confidence Level
                          </Typography>
                        </Box>
                      </Grid>

                      {/* Probability Breakdown */}
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom>
                          Probability Breakdown:
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Typography variant="body2">🌍 Confirmed</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={prediction.probabilities.confirmed}
                                sx={{ width: 100, height: 8, borderRadius: 4 }}
                              />
                              <Typography variant="body2" fontWeight="bold">
                                {prediction.probabilities.confirmed}%
                              </Typography>
                            </Box>
                          </Box>
                          
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Typography variant="body2">🔍 Candidate</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={prediction.probabilities.candidate}
                                sx={{ width: 100, height: 8, borderRadius: 4 }}
                                color="warning"
                              />
                              <Typography variant="body2" fontWeight="bold">
                                {prediction.probabilities.candidate}%
                              </Typography>
                            </Box>
                          </Box>
                          
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Typography variant="body2">❌ False Positive</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={prediction.probabilities.falsePositive}
                                sx={{ width: 100, height: 8, borderRadius: 4 }}
                                color="error"
                              />
                              <Typography variant="body2" fontWeight="bold">
                                {prediction.probabilities.falsePositive}%
                              </Typography>
                            </Box>
                          </Box>
                        </Box>

                        <Box sx={{ mt: 2, p: 2, backgroundColor: 'action.hover', borderRadius: 1 }}>
                          <Typography variant="caption" display="block" gutterBottom>
                            <strong>Uncertainty:</strong> {prediction.prediction.uncertainty}%
                          </Typography>
                          <Typography variant="caption">
                            {prediction.prediction.explanation}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Light Curve Visualization */}
          {lightCurveData && (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6">
                    📊 Light Curve Analysis
                  </Typography>
                  <Box>
                    <Tooltip title="View full screen">
                      <IconButton size="small">
                        <VisibilityIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download data">
                      <IconButton size="small">
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                
                <Box sx={{ height: 400, width: '100%' }}>
                  <Plot
                    {...createLightCurvePlot()}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                  />
                </Box>

                {/* Data Info */}
                <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Chip
                    size="small"
                    label={`${lightCurveData.lightCurveData.length} data points`}
                    icon={<TrendingUpIcon />}
                  />
                  <Chip
                    size="small"
                    label={`${selectedMission.toUpperCase()} mission`}
                    color="primary"
                  />
                  {lightCurveData.metadata?.filename && (
                    <Chip
                      size="small"
                      label={lightCurveData.metadata.filename}
                      variant="outlined"
                    />
                  )}
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default Detector;
