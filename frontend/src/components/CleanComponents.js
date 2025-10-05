import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Zoom,
  CircularProgress,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  Close as CloseIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  RocketLaunch as RocketIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { cleanCustomStyles, cleanAnimationVariants } from '../styles/cleanDarkTheme';

// Clean Black & White Card Component
export const CleanCard = ({ children, hover = true, ...props }) => {
  return (
    <motion.div
      variants={cleanAnimationVariants.fadeInScale}
      whileHover={hover ? { y: -4, scale: 1.01 } : {}}
      style={cleanCustomStyles.card}
    >
      <Card
        {...props}
        sx={{
          background: 'transparent',
          boxShadow: 'none',
          border: 'none',
          ...props.sx,
        }}
      >
        {children}
      </Card>
    </motion.div>
  );
};

// Clean Dialog Component
export const CleanDialog = ({ 
  open, 
  onClose, 
  title, 
  children, 
  actions,
  maxWidth = 'md',
  fullWidth = true,
  showCloseButton = true,
  ...props 
}) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth={maxWidth}
      fullWidth={fullWidth}
      TransitionComponent={Zoom}
      TransitionProps={{ timeout: 400 }}
      PaperProps={{
        sx: {
          background: 'rgba(20, 20, 20, 0.98)',
          backdropFilter: 'blur(30px)',
          border: '2px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '16px',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.9)',
        },
      }}
      {...props}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        color: '#ffffff',
        fontSize: '1.75rem',
        fontWeight: 700,
      }}>
        <Typography variant="h5" component="div" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
        {showCloseButton && (
          <IconButton
            onClick={onClose}
            sx={{
              color: 'rgba(255, 255, 255, 0.7)',
              '&:hover': {
                color: '#ffffff',
                background: 'rgba(255, 255, 255, 0.1)',
              },
            }}
          >
            <CloseIcon />
          </IconButton>
        )}
      </DialogTitle>
      
      <DialogContent sx={{ p: 3 }}>
        {children}
      </DialogContent>
      
      {actions && (
        <DialogActions sx={{ p: 3, pt: 0 }}>
          {actions}
        </DialogActions>
      )}
    </Dialog>
  );
};

// Clean Loading Component
export const CleanLoader = ({ 
  loading, 
  text = "Processing...", 
  subtext = "AI is analyzing your data",
  size = 60,
  showProgress = false,
  progress = 0,
}) => {
  if (!loading) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.95)',
          backdropFilter: 'blur(20px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}
      >
        <motion.div
          variants={cleanAnimationVariants.fadeInScale}
          style={{
            ...cleanCustomStyles.card,
            padding: '40px',
            textAlign: 'center',
            minWidth: '350px',
          }}
        >
          <Box sx={{ position: 'relative', display: 'inline-flex', mb: 3 }}>
            <CircularProgress
              size={size}
              thickness={3}
              sx={{
                color: '#ffffff',
                '& .MuiCircularProgress-circle': {
                  strokeLinecap: 'round',
                },
              }}
            />
            <Box
              sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <RocketIcon 
                sx={{ 
                  color: '#ffffff', 
                  fontSize: size * 0.4,
                }} 
              />
            </Box>
          </Box>
          
          <Typography 
            variant="h6" 
            sx={{ 
              mb: 1, 
              color: '#ffffff',
              fontWeight: 600,
            }}
          >
            {text}
          </Typography>
          
          <Typography 
            variant="body2" 
            sx={{ 
              color: 'rgba(255, 255, 255, 0.8)', 
              mb: 2,
            }}
          >
            {subtext}
          </Typography>
          
          {showProgress && (
            <Box sx={{ width: '100%', mt: 2 }}>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    background: '#ffffff',
                    borderRadius: 4,
                  },
                }}
              />
              <Typography 
                variant="caption" 
                sx={{ 
                  mt: 1, 
                  display: 'block',
                  color: '#ffffff',
                  fontWeight: 500,
                }}
              >
                {Math.round(progress)}% Complete
              </Typography>
            </Box>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// Clean Status Chip
export const CleanStatusChip = ({ 
  status, 
  label, 
  icon, 
  animated = true,
  ...props 
}) => {
  const getStatusConfig = (status) => {
    const baseConfig = {
      color: '#ffffff',
      background: 'rgba(255, 255, 255, 0.1)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
    };

    switch (status) {
      case 'success':
      case 'confirmed':
        return {
          ...baseConfig,
          icon: icon || <CheckIcon />,
        };
      case 'warning':
      case 'candidate':
        return {
          ...baseConfig,
          icon: icon || <WarningIcon />,
        };
      case 'error':
      case 'false_positive':
        return {
          ...baseConfig,
          icon: icon || <ErrorIcon />,
        };
      case 'info':
      default:
        return {
          ...baseConfig,
          icon: icon || <InfoIcon />,
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <motion.div
      whileHover={animated ? { scale: 1.05 } : {}}
      whileTap={animated ? { scale: 0.95 } : {}}
    >
      <Chip
        icon={config.icon}
        label={label}
        sx={{
          background: config.background,
          color: config.color,
          border: config.border,
          backdropFilter: 'blur(10px)',
          fontWeight: 500,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          fontSize: '0.75rem',
          '& .MuiChip-icon': {
            color: config.color,
          },
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.15)',
            transform: 'translateY(-1px)',
          },
          ...props.sx,
        }}
        {...props}
      />
    </motion.div>
  );
};

// Clean Metric Card
export const CleanMetricCard = ({ 
  title, 
  value, 
  subtitle, 
  icon, 
  trend,
  animated = true,
  ...props 
}) => {
  return (
    <motion.div
      variants={cleanAnimationVariants.fadeInUp}
      whileHover={animated ? { y: -4, scale: 1.01 } : {}}
    >
      <CleanCard hover={animated} {...props}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <Box sx={{ flex: 1 }}>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.8)', 
                  mb: 1,
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  fontSize: '0.75rem',
                }}
              >
                {title}
              </Typography>
              
              <Typography 
                variant="h3" 
                sx={{ 
                  fontWeight: 800,
                  color: '#ffffff',
                  mb: 0.5,
                  fontFamily: '"Inter", sans-serif',
                }}
              >
                {value}
              </Typography>
              
              {subtitle && (
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: 'rgba(255, 255, 255, 0.7)',
                    display: 'block',
                    fontWeight: 400,
                  }}
                >
                  {subtitle}
                </Typography>
              )}
              
              {trend && (
                <Box sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: '#ffffff',
                      fontWeight: 600,
                    }}
                  >
                    {trend > 0 ? '+' : ''}{trend}%
                  </Typography>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: 'rgba(255, 255, 255, 0.5)',
                      ml: 1,
                    }}
                  >
                    vs last week
                  </Typography>
                </Box>
              )}
            </Box>
            
            {icon && (
              <Box
                sx={{
                  p: 2,
                  borderRadius: '8px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  color: '#ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {React.cloneElement(icon, { 
                  sx: { 
                    fontSize: 28,
                  } 
                })}
              </Box>
            )}
          </Box>
        </CardContent>
      </CleanCard>
    </motion.div>
  );
};

// Clean Button Component
export const CleanButton = ({ 
  children, 
  variant = 'contained',
  animated = true,
  loading = false,
  ...props 
}) => {
  return (
    <motion.div
      whileHover={animated ? { scale: 1.02 } : {}}
      whileTap={animated ? { scale: 0.98 } : {}}
    >
      <Button
        variant={variant}
        disabled={loading}
        sx={{
          ...cleanCustomStyles.button,
          ...props.sx,
        }}
        {...props}
      >
        {loading ? (
          <CircularProgress 
            size={20} 
            sx={{ 
              color: '#000000',
            }} 
          />
        ) : (
          children
        )}
      </Button>
    </motion.div>
  );
};

// Clean Container with Effects
export const CleanContainer = ({ children, variant = 'default', glowColor, ...props }) => {
  const getBackgroundStyle = (variant) => {
    switch (variant) {
      case 'hero':
        return {
          background: 'rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(20px)',
        };
      case 'section':
        return cleanCustomStyles.container;
      default:
        return {};
    }
  };

  // Remove glowColor from props to prevent it from being passed to DOM
  const { sx, ...otherProps } = props;

  return (
    <Box
      sx={{
        ...getBackgroundStyle(variant),
        ...sx,
      }}
      {...otherProps}
    >
      {children}
    </Box>
  );
};

const CleanComponents = {
  CleanCard,
  CleanDialog,
  CleanLoader,
  CleanStatusChip,
  CleanMetricCard,
  CleanButton,
  CleanContainer,
};

export default CleanComponents;
