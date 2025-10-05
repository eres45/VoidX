import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import EnhancedDashboard from './pages/EnhancedDashboard';
import Detector from './pages/Detector';
import Explorer from './pages/Explorer';
import Analytics from './pages/Analytics';
import About from './pages/About';
import TestPage from './pages/TestPage';
import { cleanDarkTheme } from './styles/cleanDarkTheme';
import SpaceBackground from './components/SpaceBackground';
import './App.css';

function App() {
  return (
    <ThemeProvider theme={cleanDarkTheme}>
      <CssBaseline />
      <SpaceBackground>
        <Router>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            minHeight: '100vh',
            position: 'relative',
            zIndex: 1,
          }}>
            <Navbar />
            <Box component="main" sx={{ 
              flexGrow: 1, 
              pt: 8,
              position: 'relative',
              zIndex: 1,
            }}>
              <Routes>
                <Route path="/" element={<EnhancedDashboard />} />
                <Route path="/classic" element={<Dashboard />} />
                <Route path="/detector" element={<Detector />} />
                <Route path="/explorer" element={<Explorer />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/about" element={<About />} />
                <Route path="/test" element={<TestPage />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </SpaceBackground>
    </ThemeProvider>
  );
}

export default App;
