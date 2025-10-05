import React from 'react';
import { Container, Typography, Box, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const TestPage = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ 
        background: 'rgba(40, 40, 40, 0.8)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        p: 4,
        textAlign: 'center'
      }}>
        <Typography variant="h3" sx={{ color: '#ffffff', mb: 3 }}>
          🚀 Navigation Test Page
        </Typography>
        
        <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 4 }}>
          If you can see this page, navigation is working! Try the buttons below:
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button 
            variant="contained" 
            onClick={() => navigate('/')}
            sx={{ 
              background: '#ffffff',
              color: '#000000',
              '&:hover': { background: '#e0e0e0' }
            }}
          >
            Dashboard
          </Button>
          
          <Button 
            variant="contained" 
            onClick={() => navigate('/detector')}
            sx={{ 
              background: '#ffffff',
              color: '#000000',
              '&:hover': { background: '#e0e0e0' }
            }}
          >
            Detector
          </Button>
          
          <Button 
            variant="contained" 
            onClick={() => navigate('/explorer')}
            sx={{ 
              background: '#ffffff',
              color: '#000000',
              '&:hover': { background: '#e0e0e0' }
            }}
          >
            Explorer
          </Button>
          
          <Button 
            variant="contained" 
            onClick={() => navigate('/analytics')}
            sx={{ 
              background: '#ffffff',
              color: '#000000',
              '&:hover': { background: '#e0e0e0' }
            }}
          >
            Analytics
          </Button>
          
          <Button 
            variant="contained" 
            onClick={() => navigate('/about')}
            sx={{ 
              background: '#ffffff',
              color: '#000000',
              '&:hover': { background: '#e0e0e0' }
            }}
          >
            About
          </Button>
        </Box>
        
        <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)', mt: 4 }}>
          Navigation Status: ✅ Working
        </Typography>
      </Box>
    </Container>
  );
};

export default TestPage;
