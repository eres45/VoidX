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
import { darkCustomStyles, darkAnimationVariants } from '../styles/premiumDarkTheme';

// Premium Dark Neon Card Component
export const NeonCard = ({ children, hover = true, glow = false, glowColor = '#00ffff', ...props }) => {
  return (
    <motion.div
      variants={darkAnimationVariants.fadeInScale}
      whileHover={hover ? { y: -8, scale: 1.02 } : {}}
      style={{
        ...darkCustomStyles.neonCard,
        ...(glow && {
          boxShadow: `0 8px 32px rgba(0, 0, 0, 0.6), 0 0 40px ${glowColor}40`,
        }),
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

// Premium Dark Dialog Component
export const NeonDialog = ({ 
  open, 
  onClose, 
  title, 
  children, 
  actions,
  maxWidth = 'md',
  fullWidth = true,
  showCloseButton = true,
  glowColor = '#00ffff',
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
          background: 'rgba(5, 5, 5, 0.98)',
          backdropFilter: 'blur(30px)',
          border: `2px solid ${glowColor}50`,
          borderRadius: '20px',
          boxShadow: `0 20px 60px rgba(0, 0, 0, 0.9), 0 0 100px ${glowColor}30`,
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: -2,
            left: -2,
            right: -2,
            bottom: -2,
            borderRadius: 22,
            background: `linear-gradient(45deg, ${glowColor}, #ff0080, #00ff88, #0099ff)`,
            zIndex: -1,
            opacity: 0.3,
          },
        },
      }}
      {...props}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        ...darkCustomStyles.neonText,
        fontSize: '1.75rem',
        fontWeight: 800,
        textShadow: `0 0 20px ${glowColor}80`,
      }}>
        <Typography variant="h5" component="div" sx={{ fontWeight: 800 }}>
          {title}
        </Typography>
        {showCloseButton && (
          <IconButton
            onClick={onClose}
            sx={{
              color: 'rgba(255, 255, 255, 0.7)',
              '&:hover': {
                color: glowColor,
                background: `${glowColor}20`,
                boxShadow: `0 0 20px ${glowColor}40`,
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

// Premium Dark Loading Component
export const NeonLoader = ({ 
  loading, 
  text = "Processing...", 
  subtext = "AI is analyzing your data",
  size = 60,
  showProgress = false,
  progress = 0,
  glowColor = '#00ffff',
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
          variants={darkAnimationVariants.fadeInScale}
          style={{
            ...darkCustomStyles.neonCard,
            padding: '40px',
            textAlign: 'center',
            minWidth: '350px',
            boxShadow: `0 20px 60px rgba(0, 0, 0, 0.8), 0 0 80px ${glowColor}30`,
          }}
        >
          <Box sx={{ position: 'relative', display: 'inline-flex', mb: 3 }}>
            <CircularProgress
              size={size}
              thickness={3}
              sx={{
                color: glowColor,
                filter: `drop-shadow(0 0 10px ${glowColor})`,
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
                  color: '#ff0080', 
                  fontSize: size * 0.4,
                  filter: 'drop-shadow(0 0 8px #ff0080)',
                }} 
              />
            </Box>
          </Box>
          
          <Typography 
            variant="h6" 
            sx={{ 
              mb: 1, 
              ...darkCustomStyles.neonText,
              textShadow: `0 0 15px ${glowColor}60`,
            }}
          >
            {text}
          </Typography>
          
          <Typography 
            variant="body2" 
            sx={{ 
              color: 'rgba(255, 255, 255, 0.8)', 
              mb: 2,
              textShadow: '0 0 10px rgba(255, 255, 255, 0.3)',
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
                  height: 10,
                  borderRadius: 5,
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    background: `linear-gradient(90deg, ${glowColor} 0%, #ff0080 50%, #00ff88 100%)`,
                    borderRadius: 5,
                    boxShadow: `0 0 15px ${glowColor}50`,
                  },
                }}
              />
              <Typography 
                variant="caption" 
                sx={{ 
                  mt: 1, 
                  display: 'block',
                  color: glowColor,
                  fontWeight: 600,
                  textShadow: `0 0 10px ${glowColor}50`,
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

// Premium Dark Status Chip
export const NeonStatusChip = ({ 
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
          color: '#00ff88',
          background: 'rgba(0, 255, 136, 0.2)',
          border: '1px solid rgba(0, 255, 136, 0.4)',
          glow: '0 0 20px rgba(0, 255, 136, 0.3)',
          icon: icon || <CheckIcon />,
        };
      case 'warning':
      case 'candidate':
        return {
          color: '#ffaa00',
          background: 'rgba(255, 170, 0, 0.2)',
          border: '1px solid rgba(255, 170, 0, 0.4)',
          glow: '0 0 20px rgba(255, 170, 0, 0.3)',
          icon: icon || <WarningIcon />,
        };
      case 'error':
      case 'false_positive':
        return {
          color: '#ff3366',
          background: 'rgba(255, 51, 102, 0.2)',
          border: '1px solid rgba(255, 51, 102, 0.4)',
          glow: '0 0 20px rgba(255, 51, 102, 0.3)',
          icon: icon || <ErrorIcon />,
        };
      case 'info':
      default:
        return {
          color: '#00ffff',
          background: 'rgba(0, 255, 255, 0.2)',
          border: '1px solid rgba(0, 255, 255, 0.4)',
          glow: '0 0 20px rgba(0, 255, 255, 0.3)',
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
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          boxShadow: config.glow,
          '& .MuiChip-icon': {
            color: config.color,
            filter: `drop-shadow(0 0 5px ${config.color})`,
          },
          '&:hover': {
            boxShadow: `${config.glow}, 0 0 30px ${config.color}40`,
            transform: 'translateY(-1px)',
          },
          ...props.sx,
        }}
        {...props}
      />
    </motion.div>
  );
};

// Premium Dark Metric Card
export const NeonMetricCard = ({ 
  title, 
  value, 
  subtitle, 
  icon, 
  color = '#00ffff',
  trend,
  animated = true,
  ...props 
}) => {
  return (
    <motion.div
      variants={darkAnimationVariants.fadeInUp}
      whileHover={animated ? { y: -8, scale: 1.02 } : {}}
    >
      <NeonCard hover={animated} glowColor={color} {...props}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <Box sx={{ flex: 1 }}>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.8)', 
                  mb: 1,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                }}
              >
                {title}
              </Typography>
              
              <Typography 
                variant="h3" 
                sx={{ 
                  fontWeight: 900,
                  color: color,
                  mb: 0.5,
                  textShadow: `0 0 20px ${color}60`,
                  fontFamily: '"JetBrains Mono", monospace',
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
                    fontWeight: 500,
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
                      color: trend > 0 ? '#00ff88' : '#ff3366',
                      fontWeight: 700,
                      textShadow: trend > 0 ? '0 0 10px #00ff88' : '0 0 10px #ff3366',
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
                  borderRadius: '12px',
                  background: `${color}20`,
                  border: `1px solid ${color}40`,
                  color: color,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: `0 0 20px ${color}30`,
                }}
              >
                {React.cloneElement(icon, { 
                  sx: { 
                    fontSize: 32,
                    filter: `drop-shadow(0 0 8px ${color})`,
                  } 
                })}
              </Box>
            )}
          </Box>
        </CardContent>
      </NeonCard>
    </motion.div>
  );
};

// Premium Dark Button Component
export const NeonButton = ({ 
  children, 
  variant = 'contained',
  color = '#00ffff',
  glow = true,
  animated = true,
  loading = false,
  ...props 
}) => {
  return (
    <motion.div
      whileHover={animated ? { scale: 1.05 } : {}}
      whileTap={animated ? { scale: 0.95 } : {}}
    >
      <Button
        variant={variant}
        disabled={loading}
        sx={{
          ...darkCustomStyles.glowButton,
          background: `linear-gradient(45deg, ${color} 0%, ${color}cc 100%)`,
          border: `1px solid ${color}80`,
          boxShadow: glow ? `0 0 20px ${color}40` : 'none',
          '&:hover': {
            background: `linear-gradient(45deg, ${color}ff 0%, ${color} 100%)`,
            boxShadow: glow ? `0 0 30px ${color}60` : 'none',
            transform: 'translateY(-2px)',
          },
          '&:active': {
            transform: 'translateY(0)',
          },
          ...props.sx,
        }}
        {...props}
      >
        {loading ? (
          <CircularProgress 
            size={20} 
            sx={{ 
              color: '#000000',
              filter: 'drop-shadow(0 0 5px #000000)',
            }} 
          />
        ) : (
          children
        )}
      </Button>
    </motion.div>
  );
};

// Premium Dark Container with Neon Effects
export const NeonContainer = ({ children, variant = 'default', glowColor = '#00ffff', ...props }) => {
  const getBackgroundStyle = (variant) => {
    switch (variant) {
      case 'hero':
        return {
          background: `radial-gradient(ellipse at center, ${glowColor}10 0%, transparent 70%)`,
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `
              radial-gradient(circle at 20% 20%, ${glowColor}15 0%, transparent 50%),
              radial-gradient(circle at 80% 80%, #ff008015 0%, transparent 50%),
              radial-gradient(circle at 40% 60%, #00ff8810 0%, transparent 50%)
            `,
            pointerEvents: 'none',
            zIndex: -1,
          },
        };
      case 'section':
        return {
          background: 'rgba(0, 0, 0, 0.6)',
          backdropFilter: 'blur(20px)',
          border: `1px solid ${glowColor}20`,
          borderRadius: '20px',
          boxShadow: `0 8px 32px rgba(0, 0, 0, 0.6), 0 0 40px ${glowColor}10`,
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

const DarkPremiumComponents = {
  NeonCard,
  NeonDialog,
  NeonLoader,
  NeonStatusChip,
  NeonMetricCard,
  NeonButton,
  NeonContainer,
};

export default DarkPremiumComponents;
