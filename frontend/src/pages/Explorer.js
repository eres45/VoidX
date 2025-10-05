import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  TextField,
  InputAdornment,
  Chip,
  Avatar,
  IconButton,
  Pagination,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Star as StarIcon,
  Public as PublicIcon,
  Satellite as SatelliteIcon,
  Science as ScienceIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';

const Explorer = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedMission, setSelectedMission] = useState('all');
  const [selectedDisposition, setSelectedDisposition] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedObject, setSelectedObject] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(false);

  // Mock data for demonstration
  const mockObjects = [
    {
      id: 'KOI-326.01',
      name: 'Kepler-22b',
      mission: 'Kepler',
      disposition: 'CONFIRMED',
      period: 289.86,
      radius: 2.38,
      temperature: 262,
      distance: 620,
      host_star: 'Kepler-22',
      discovery_year: 2011,
      description: 'First confirmed planet in the habitable zone of a Sun-like star',
      lightCurve: Array.from({length: 1000}, (_, i) => 1 + 0.01 * Math.sin(0.02 * i) + 0.005 * Math.random() - (i % 200 < 10 ? 0.05 : 0))
    },
    {
      id: 'TOI-715.01',
      name: 'TOI-715 b',
      mission: 'TESS',
      disposition: 'CONFIRMED',
      period: 19.28,
      radius: 1.55,
      temperature: 347,
      distance: 137,
      host_star: 'TOI-715',
      discovery_year: 2024,
      description: 'Super-Earth in the habitable zone discovered by TESS',
      lightCurve: Array.from({length: 800}, (_, i) => 1 + 0.02 * Math.sin(0.03 * i) + 0.008 * Math.random() - (i % 120 < 8 ? 0.03 : 0))
    },
    {
      id: 'K2-18b',
      name: 'K2-18 b',
      mission: 'K2',
      disposition: 'CONFIRMED',
      period: 32.94,
      radius: 2.23,
      temperature: 279,
      distance: 124,
      host_star: 'K2-18',
      discovery_year: 2015,
      description: 'Sub-Neptune with potential water vapor in atmosphere',
      lightCurve: Array.from({length: 900}, (_, i) => 1 + 0.015 * Math.sin(0.025 * i) + 0.01 * Math.random() - (i % 180 < 12 ? 0.04 : 0))
    },
    {
      id: 'KOI-5715.01',
      name: 'KOI-5715.01',
      mission: 'Kepler',
      disposition: 'CANDIDATE',
      period: 188.75,
      radius: 1.89,
      temperature: 301,
      distance: 892,
      host_star: 'KOI-5715',
      discovery_year: 2023,
      description: 'Promising Earth-sized candidate awaiting confirmation',
      lightCurve: Array.from({length: 1200}, (_, i) => 1 + 0.008 * Math.sin(0.015 * i) + 0.012 * Math.random() - (i % 250 < 6 ? 0.02 : 0))
    },
    {
      id: 'TOI-2109.01',
      name: 'TOI-2109 b',
      mission: 'TESS',
      disposition: 'CONFIRMED',
      period: 0.67,
      radius: 1.35,
      temperature: 3000,
      distance: 855,
      host_star: 'TOI-2109',
      discovery_year: 2021,
      description: 'Ultra-hot Jupiter with the shortest known period',
      lightCurve: Array.from({length: 600}, (_, i) => 1 + 0.03 * Math.sin(0.1 * i) + 0.006 * Math.random() - (i % 40 < 5 ? 0.08 : 0))
    },
    {
      id: 'EPIC-201367065.01',
      name: 'K2-138 g',
      mission: 'K2',
      disposition: 'CANDIDATE',
      period: 41.97,
      radius: 3.44,
      temperature: 254,
      distance: 792,
      host_star: 'K2-138',
      discovery_year: 2018,
      description: 'Part of a resonant chain system discovered by citizen scientists',
      lightCurve: Array.from({length: 950}, (_, i) => 1 + 0.012 * Math.sin(0.02 * i) + 0.009 * Math.random() - (i % 220 < 9 ? 0.035 : 0))
    }
  ];

  useEffect(() => {
    setObjects(mockObjects);
  }, []);

  // Filter objects based on search and filters
  const filteredObjects = objects.filter(obj => {
    const matchesSearch = obj.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         obj.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         obj.host_star.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesMission = selectedMission === 'all' || obj.mission.toLowerCase() === selectedMission;
    const matchesDisposition = selectedDisposition === 'all' || obj.disposition === selectedDisposition;
    
    return matchesSearch && matchesMission && matchesDisposition;
  });

  // Pagination
  const itemsPerPage = 6;
  const totalPages = Math.ceil(filteredObjects.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedObjects = filteredObjects.slice(startIndex, startIndex + itemsPerPage);

  const handleViewObject = (object) => {
    setSelectedObject(object);
    setDialogOpen(true);
  };

  const getMissionColor = (mission) => {
    switch (mission.toLowerCase()) {
      case 'kepler': return '#00a8cc';
      case 'k2': return '#ff6b35';
      case 'tess': return '#00ff88';
      default: return '#b0b0b0';
    }
  };

  const getDispositionChip = (disposition) => {
    const config = {
      'CONFIRMED': { color: 'success', icon: '✅' },
      'CANDIDATE': { color: 'warning', icon: '🔍' },
      'FALSE_POSITIVE': { color: 'error', icon: '❌' }
    };
    
    const { color, icon } = config[disposition] || config['CANDIDATE'];
    
    return (
      <Chip
        label={`${icon} ${disposition}`}
        color={color}
        size="small"
        variant="outlined"
      />
    );
  };

  const createLightCurvePlot = (lightCurve, objectName) => ({
    data: [{
      x: Array.from({length: lightCurve.length}, (_, i) => i),
      y: lightCurve,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#00a8cc', width: 1 },
      name: 'Flux'
    }],
    layout: {
      title: {
        text: `Light Curve - ${objectName}`,
        font: { color: '#ffffff', size: 14 },
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
      margin: { l: 50, r: 20, t: 40, b: 40 },
      height: 300,
    },
    config: {
      displayModeBar: false,
    },
  });

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          🌌 Exoplanet Explorer
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Browse and analyze exoplanets discovered by NASA's space missions
        </Typography>
      </Box>

      {/* Search and Filters */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                placeholder="Search by name, ID, or host star..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Mission</InputLabel>
                <Select
                  value={selectedMission}
                  label="Mission"
                  onChange={(e) => setSelectedMission(e.target.value)}
                >
                  <MenuItem value="all">All Missions</MenuItem>
                  <MenuItem value="kepler">Kepler</MenuItem>
                  <MenuItem value="k2">K2</MenuItem>
                  <MenuItem value="tess">TESS</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={selectedDisposition}
                  label="Status"
                  onChange={(e) => setSelectedDisposition(e.target.value)}
                >
                  <MenuItem value="all">All Status</MenuItem>
                  <MenuItem value="CONFIRMED">Confirmed</MenuItem>
                  <MenuItem value="CANDIDATE">Candidate</MenuItem>
                  <MenuItem value="FALSE_POSITIVE">False Positive</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={2}>
              <Typography variant="body2" color="text.secondary" align="center">
                {filteredObjects.length} objects found
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Objects Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {paginatedObjects.map((object, index) => (
          <Grid item xs={12} md={6} lg={4} key={object.id}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1 }}>
                  {/* Header */}
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Avatar
                      sx={{
                        bgcolor: getMissionColor(object.mission),
                        width: 32,
                        height: 32,
                        fontSize: '0.8rem'
                      }}
                    >
                      {object.mission[0]}
                    </Avatar>
                    {getDispositionChip(object.disposition)}
                  </Box>

                  {/* Object Info */}
                  <Typography variant="h6" gutterBottom noWrap>
                    {object.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                    ID: {object.id}
                  </Typography>

                  {/* Key Properties */}
                  <Box sx={{ my: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">Period:</Typography>
                      <Typography variant="body2">{object.period.toFixed(2)} days</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">Radius:</Typography>
                      <Typography variant="body2">{object.radius.toFixed(2)} R⊕</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">Temperature:</Typography>
                      <Typography variant="body2">{object.temperature} K</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">Distance:</Typography>
                      <Typography variant="body2">{object.distance} ly</Typography>
                    </Box>
                  </Box>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {object.description}
                  </Typography>

                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                    <Chip label={object.mission} size="small" />
                    <Chip label={`${object.discovery_year}`} size="small" variant="outlined" />
                    <Chip label={object.host_star} size="small" variant="outlined" />
                  </Box>
                </CardContent>

                {/* Actions */}
                <Box sx={{ p: 2, pt: 0 }}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<ViewIcon />}
                    onClick={() => handleViewObject(object)}
                  >
                    View Details
                  </Button>
                </Box>
              </Card>
            </motion.div>
          </Grid>
        ))}
      </Grid>

      {/* Pagination */}
      {totalPages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
          <Pagination
            count={totalPages}
            page={currentPage}
            onChange={(e, page) => setCurrentPage(page)}
            color="primary"
            size="large"
          />
        </Box>
      )}

      {/* Object Detail Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { backgroundColor: 'background.paper' }
        }}
      >
        {selectedObject && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <ScienceIcon color="primary" />
                <Box>
                  <Typography variant="h6">{selectedObject.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {selectedObject.id} • {selectedObject.mission} Mission
                  </Typography>
                </Box>
              </Box>
            </DialogTitle>
            
            <DialogContent>
              <Grid container spacing={3}>
                {/* Properties */}
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Planetary Properties
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Orbital Period:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {selectedObject.period.toFixed(2)} days
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Planet Radius:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {selectedObject.radius.toFixed(2)} R⊕
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Equilibrium Temperature:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {selectedObject.temperature} K
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Distance from Earth:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {selectedObject.distance} light-years
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Host Star:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {selectedObject.host_star}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Discovery Year:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {selectedObject.discovery_year}
                      </Typography>
                    </Box>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Status:</Typography>
                      {getDispositionChip(selectedObject.disposition)}
                    </Box>
                  </Box>

                  <Box sx={{ mt: 3 }}>
                    <Typography variant="body2" color="text.secondary">
                      {selectedObject.description}
                    </Typography>
                  </Box>
                </Grid>

                {/* Light Curve */}
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Light Curve Data
                  </Typography>
                  <Box sx={{ height: 300, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 1, p: 1 }}>
                    <Plot
                      {...createLightCurvePlot(selectedObject.lightCurve, selectedObject.name)}
                      style={{ width: '100%', height: '100%' }}
                      useResizeHandler
                    />
                  </Box>
                </Grid>
              </Grid>
            </DialogContent>

            <DialogActions>
              <Button startIcon={<DownloadIcon />} color="primary">
                Download Data
              </Button>
              <Button onClick={() => setDialogOpen(false)}>
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Container>
  );
};

export default Explorer;
