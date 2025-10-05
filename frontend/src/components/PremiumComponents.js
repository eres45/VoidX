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
  Fade,
  Slide,
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
import { customStyles, animationVariants } from '../styles/premiumTheme';

// Premium Glassmorphism Card Component
export const GlassCard = ({ children, hover = true, glow = false, ...props }) => {
  return (
    <motion.div
      variants={animationVariants.fadeInScale}
      whileHover={hover ? { y: -8, scale: 1.02 } : {}}
      style={{
        ...customStyles.glassmorphism,
        ...(glow && customStyles.neonGlow),
      }}
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

// Premium Modal/Dialog Component
export const PremiumDialog = ({ 
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
          background: 'rgba(26, 26, 42, 0.95)',
          backdropFilter: 'blur(30px)',
          border: '1px solid rgba(0, 212, 255, 0.3)',
          borderRadius: '24px',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
        },
      }}
      {...props}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        background: 'linear-gradient(135deg, #00d4ff 0%, #ff6b35 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        backgroundClip: 'text',
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
                color: '#00d4ff',
                background: 'rgba(0, 212, 255, 0.1)',
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

// Premium Loading Component
export const PremiumLoader = ({ 
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
          background: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(10px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}
      >
        <motion.div
          variants={animationVariants.fadeInScale}
          style={{
            ...customStyles.glassmorphism,
            padding: '40px',
            textAlign: 'center',
            minWidth: '300px',
          }}
        >
          <Box sx={{ position: 'relative', display: 'inline-flex', mb: 3 }}>
            <CircularProgress
              size={size}
              thickness={4}
              sx={{
                color: '#00d4ff',
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
              <RocketIcon sx={{ color: '#ff6b35', fontSize: size * 0.4 }} />
            </Box>
          </Box>
          
          <Typography variant="h6" sx={{ mb: 1, ...customStyles.gradientText }}>
            {text}
          </Typography>
          
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 2 }}>
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
                    background: 'linear-gradient(90deg, #00d4ff 0%, #ff6b35 100%)',
                    borderRadius: 4,
                  },
                }}
              />
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                {Math.round(progress)}% Complete
              </Typography>
            </Box>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// Premium Status Chip
export const StatusChip = ({ 
  status, 
  label, 
  icon, 
  animated = true,
  ...props 
}) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'success':
      case 'confirmed':
        return {
          color: '#2ed573',
          background: 'rgba(46, 213, 115, 0.2)',
          icon: icon || <CheckIcon />,
        };
      case 'warning':
      case 'candidate':
        return {
          color: '#ffa502',
          background: 'rgba(255, 165, 2, 0.2)',
          icon: icon || <WarningIcon />,
        };
      case 'error':
      case 'false_positive':
        return {
          color: '#ff4757',
          background: 'rgba(255, 71, 87, 0.2)',
          icon: icon || <ErrorIcon />,
        };
      case 'info':
      default:
        return {
          color: '#00d4ff',
          background: 'rgba(0, 212, 255, 0.2)',
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
          border: `1px solid ${config.color}40`,
          backdropFilter: 'blur(10px)',
          fontWeight: 600,
          '& .MuiChip-icon': {
            color: config.color,
          },
          ...props.sx,
        }}
        {...props}
      />
    </motion.div>
  );
};

// Premium Metric Card
export const MetricCard = ({ 
  title, 
  value, 
  subtitle, 
  icon, 
  color = '#00d4ff',
  trend,
  animated = true,
  ...props 
}) => {
  return (
    <motion.div
      variants={animationVariants.fadeInUp}
      whileHover={animated ? animationVariants.floatingCard.animate : {}}
    >
      <GlassCard hover={animated} {...props}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <Box sx={{ flex: 1 }}>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.7)', 
                  mb: 1,
                  fontWeight: 500,
                }}
              >
                {title}
              </Typography>
              
              <Typography 
                variant="h3" 
                sx={{ 
                  fontWeight: 800,
                  color: color,
                  mb: 0.5,
                  background: `linear-gradient(135deg, ${color} 0%, ${color}80 100%)`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                {value}
              </Typography>
              
              {subtitle && (
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: 'rgba(255, 255, 255, 0.6)',
                    display: 'block',
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
                      color: trend > 0 ? '#2ed573' : '#ff4757',
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
                  p: 1.5,
                  borderRadius: '12px',
                  background: `${color}20`,
                  color: color,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {React.cloneElement(icon, { sx: { fontSize: 28 } })}
              </Box>
            )}
          </Box>
        </CardContent>
      </GlassCard>
    </motion.div>
  );
};

// Premium Button Component
export const PremiumButton = ({ 
  children, 
  variant = 'contained',
  color = 'primary',
  glow = false,
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
        color={color}
        disabled={loading}
        sx={{
          position: 'relative',
          overflow: 'hidden',
          ...(glow && {
            boxShadow: '0 0 20px rgba(0, 212, 255, 0.4)',
            '&:hover': {
              boxShadow: '0 0 30px rgba(0, 212, 255, 0.6)',
            },
          }),
          '&::before': glow ? {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
            transition: 'left 0.5s',
          } : {},
          '&:hover::before': glow ? {
            left: '100%',
          } : {},
          ...props.sx,
        }}
        {...props}
      >
        {loading ? (
          <CircularProgress size={20} sx={{ color: 'inherit' }} />
        ) : (
          children
        )}
      </Button>
    </motion.div>
  );
};

// Premium Container with Background Effects
export const PremiumContainer = ({ children, variant = 'default', ...props }) => {
  const getBackgroundStyle = (variant) => {
    switch (variant) {
      case 'hero':
        return {
          background: 'radial-gradient(ellipse at center, rgba(0, 212, 255, 0.1) 0%, transparent 70%)',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%2300d4ff" fill-opacity="0.05"%3E%3Ccircle cx="30" cy="30" r="1"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
            opacity: 0.3,
          },
        };
      case 'section':
        return {
          background: 'rgba(255, 255, 255, 0.02)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          borderRadius: '20px',
        };
      default:
        return {};
    }
  };

  return (
    <Box
      sx={{
        ...getBackgroundStyle(variant),
        ...props.sx,
      }}
      {...props}
    >
      {children}
    </Box>
  );
};

export default {
  GlassCard,
  PremiumDialog,
  PremiumLoader,
  StatusChip,
  MetricCard,
  PremiumButton,
  PremiumContainer,
};
