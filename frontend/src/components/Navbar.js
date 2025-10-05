import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Search as SearchIcon,
  Explore as ExploreIcon,
  Analytics as AnalyticsIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const Navbar = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const navItems = [
    { name: 'Dashboard', path: '/', icon: <DashboardIcon /> },
    { name: 'Detector', path: '/detector', icon: <SearchIcon /> },
    { name: 'Explorer', path: '/explorer', icon: <ExploreIcon /> },
    { name: 'Analytics', path: '/analytics', icon: <AnalyticsIcon /> },
    { name: 'About', path: '/about', icon: <InfoIcon /> },
  ];

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavigation = (path) => {
    navigate(path);
    setMobileOpen(false);
  };

  const drawer = (
    <Box sx={{ width: 250, mt: 2 }}>
      <List>
        {navItems.map((item) => (
          <ListItem
            button
            key={item.name}
            onClick={() => handleNavigation(item.path)}
            sx={{
              mb: 1,
              mx: 2,
              borderRadius: 2,
              backgroundColor: location.pathname === item.path ? 'primary.main' : 'transparent',
              '&:hover': {
                backgroundColor: location.pathname === item.path ? 'primary.dark' : 'rgba(255, 255, 255, 0.08)',
              },
            }}
          >
            <ListItemIcon sx={{ color: 'inherit' }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.name} />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      <AppBar position="fixed" elevation={0} sx={{ zIndex: 1100 }}>
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            style={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}
          >
            <img 
              src="/logo.png" 
              alt="ExoAI Hunter Logo" 
              style={{ 
                width: 96, 
                height: 96, 
                marginRight: 16,
                objectFit: 'contain'
              }} 
            />
            <Typography variant="h6" component="div" sx={{ fontWeight: 700 }}>
              ExoAI Hunter
            </Typography>
            <Typography variant="body2" sx={{ ml: 2, opacity: 0.7, display: { xs: 'none', sm: 'block' } }}>
              AI-Powered Exoplanet Detection
            </Typography>
          </motion.div>

          {!isMobile && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {navItems.map((item) => (
                <motion.div
                  key={item.name}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    color="inherit"
                    onClick={() => handleNavigation(item.path)}
                    startIcon={item.icon}
                    sx={{
                      mx: 0.5,
                      px: 2,
                      py: 1,
                      borderRadius: 2,
                      backgroundColor: location.pathname === item.path ? 'primary.main' : 'transparent',
                      '&:hover': {
                        backgroundColor: location.pathname === item.path ? 'primary.dark' : 'rgba(255, 255, 255, 0.08)',
                      },
                    }}
                  >
                    {item.name}
                  </Button>
                </motion.div>
              ))}
            </Box>
          )}
        </Toolbar>
      </AppBar>

      <Box component="nav">
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: 250,
              backgroundColor: 'background.paper',
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>
    </>
  );
};

export default Navbar;
