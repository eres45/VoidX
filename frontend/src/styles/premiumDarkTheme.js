import { createTheme } from '@mui/material/styles';

// Premium Dark Theme with Glow Effects - Professional & Clean
export const premiumDarkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ffff', // Bright cyan glow
      light: '#4dffff',
      dark: '#00cccc',
      contrastText: '#000000',
    },
    secondary: {
      main: '#ff0080', // Neon pink accent
      light: '#ff4da6',
      dark: '#cc0066',
      contrastText: '#ffffff',
    },
    background: {
      default: '#000000', // Pure black background
      paper: 'rgba(10, 10, 10, 0.95)', // Very dark with slight transparency
    },
    surface: {
      main: 'rgba(20, 20, 20, 0.8)',
      light: 'rgba(30, 30, 30, 0.9)',
      dark: 'rgba(5, 5, 5, 0.95)',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.9)',
      disabled: 'rgba(255, 255, 255, 0.4)',
    },
    error: {
      main: '#ff3366',
      light: '#ff6699',
      dark: '#cc1a4d',
      glow: '0 0 20px #ff3366',
    },
    warning: {
      main: '#ffaa00',
      light: '#ffcc4d',
      dark: '#cc8800',
      glow: '0 0 20px #ffaa00',
    },
    success: {
      main: '#00ff88',
      light: '#4dffaa',
      dark: '#00cc6a',
      glow: '0 0 20px #00ff88',
    },
    info: {
      main: '#0099ff',
      light: '#4dbbff',
      dark: '#0077cc',
      glow: '0 0 20px #0099ff',
    },
    // Custom glow colors
    glow: {
      cyan: '#00ffff',
      pink: '#ff0080',
      green: '#00ff88',
      blue: '#0099ff',
      purple: '#8800ff',
      orange: '#ff6600',
    },
  },
  typography: {
    fontFamily: '"JetBrains Mono", "Fira Code", "SF Mono", "Monaco", "Cascadia Code", monospace',
    h1: {
      fontSize: '3.5rem',
      fontWeight: 900,
      letterSpacing: '-0.02em',
      background: 'linear-gradient(45deg, #00ffff 0%, #ff0080 50%, #00ff88 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
      textShadow: '0 0 30px rgba(0, 255, 255, 0.5)',
    },
    h2: {
      fontSize: '2.75rem',
      fontWeight: 800,
      letterSpacing: '-0.01em',
      background: 'linear-gradient(135deg, #00ffff 0%, #0099ff 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
      textShadow: '0 0 20px rgba(0, 255, 255, 0.3)',
    },
    h3: {
      fontSize: '2.25rem',
      fontWeight: 700,
      letterSpacing: '-0.01em',
      color: '#00ffff',
      textShadow: '0 0 15px rgba(0, 255, 255, 0.4)',
    },
    h4: {
      fontSize: '1.75rem',
      fontWeight: 600,
      color: '#ffffff',
      textShadow: '0 0 10px rgba(255, 255, 255, 0.2)',
    },
    h5: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#00ffff',
    },
    h6: {
      fontSize: '1.25rem',
      fontWeight: 500,
      color: '#ffffff',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.7,
      fontWeight: 400,
      color: 'rgba(255, 255, 255, 0.9)',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      fontWeight: 400,
      color: 'rgba(255, 255, 255, 0.8)',
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 700,
      textTransform: 'uppercase',
      letterSpacing: '0.1em',
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    // Standard shadows with glow
    '0 2px 8px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 255, 255, 0.1)',
    '0 4px 16px rgba(0, 0, 0, 0.4), 0 0 30px rgba(0, 255, 255, 0.15)',
    '0 8px 24px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 255, 255, 0.2)',
    '0 12px 32px rgba(0, 0, 0, 0.6), 0 0 50px rgba(0, 255, 255, 0.25)',
    '0 16px 40px rgba(0, 0, 0, 0.7), 0 0 60px rgba(0, 255, 255, 0.3)',
    // Premium glow shadows
    '0 8px 32px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 255, 255, 0.4)',
    '0 12px 40px rgba(0, 0, 0, 0.6), 0 0 50px rgba(255, 0, 128, 0.3)',
    '0 16px 48px rgba(0, 0, 0, 0.7), 0 0 60px rgba(0, 255, 136, 0.3)',
    // Ultra premium shadows
    '0 20px 60px rgba(0, 0, 0, 0.8), 0 0 80px rgba(0, 255, 255, 0.4)',
    '0 24px 80px rgba(0, 0, 0, 0.9), 0 0 100px rgba(255, 0, 128, 0.4)',
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: '#000000',
          backgroundAttachment: 'fixed',
          minHeight: '100vh',
          overflow: 'hidden',
        },
        '*': {
          scrollbarWidth: 'thin',
          scrollbarColor: '#00ffff #1a1a1a',
        },
        '*::-webkit-scrollbar': {
          width: '8px',
        },
        '*::-webkit-scrollbar-track': {
          background: '#1a1a1a',
        },
        '*::-webkit-scrollbar-thumb': {
          background: 'linear-gradient(45deg, #00ffff, #ff0080)',
          borderRadius: '4px',
          '&:hover': {
            background: 'linear-gradient(45deg, #4dffff, #ff4da6)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'uppercase',
          fontSize: '0.875rem',
          fontWeight: 700,
          padding: '12px 32px',
          letterSpacing: '0.1em',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
            transition: 'left 0.5s',
          },
          '&:hover': {
            transform: 'translateY(-2px) scale(1.02)',
            '&::before': {
              left: '100%',
            },
          },
          '&:active': {
            transform: 'translateY(0) scale(0.98)',
          },
        },
        contained: {
          background: 'linear-gradient(45deg, #00ffff 0%, #0099ff 100%)',
          color: '#000000',
          boxShadow: '0 4px 20px rgba(0, 255, 255, 0.4), 0 0 40px rgba(0, 255, 255, 0.2)',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          '&:hover': {
            background: 'linear-gradient(45deg, #4dffff 0%, #4dbbff 100%)',
            boxShadow: '0 6px 30px rgba(0, 255, 255, 0.6), 0 0 60px rgba(0, 255, 255, 0.3)',
          },
        },
        outlined: {
          borderColor: '#00ffff',
          color: '#00ffff',
          background: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
          '&:hover': {
            borderColor: '#4dffff',
            background: 'rgba(0, 255, 255, 0.1)',
            boxShadow: '0 0 30px rgba(0, 255, 255, 0.4)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          background: 'rgba(10, 10, 10, 0.8)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6), 0 0 40px rgba(0, 255, 255, 0.1)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            borderRadius: 16,
            padding: '1px',
            background: 'linear-gradient(45deg, rgba(0, 255, 255, 0.3), rgba(255, 0, 128, 0.3), rgba(0, 255, 136, 0.3))',
            mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
            maskComposite: 'exclude',
            WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
            WebkitMaskComposite: 'xor',
            opacity: 0,
            transition: 'opacity 0.3s ease',
          },
          '&:hover': {
            transform: 'translateY(-8px) scale(1.02)',
            boxShadow: '0 16px 48px rgba(0, 0, 0, 0.8), 0 0 60px rgba(0, 255, 255, 0.3)',
            border: '1px solid rgba(0, 255, 255, 0.4)',
            '&::before': {
              opacity: 1,
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'rgba(10, 10, 10, 0.9)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(0, 255, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(0, 0, 0, 0.95)',
          backdropFilter: 'blur(20px)',
          border: 'none',
          borderBottom: '1px solid rgba(0, 255, 255, 0.2)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.8), 0 0 40px rgba(0, 255, 255, 0.1)',
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 20,
          background: 'rgba(5, 5, 5, 0.98)',
          backdropFilter: 'blur(30px)',
          border: '2px solid rgba(0, 255, 255, 0.3)',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.9), 0 0 100px rgba(0, 255, 255, 0.2)',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: -2,
            left: -2,
            right: -2,
            bottom: -2,
            borderRadius: 22,
            background: 'linear-gradient(45deg, #00ffff, #ff0080, #00ff88, #0099ff)',
            zIndex: -1,
            opacity: 0.3,
          },
        },
      },
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(45deg, #00ffff 0%, #ff0080 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontSize: '1.75rem',
          fontWeight: 800,
          padding: '24px 24px 16px',
          textShadow: '0 0 20px rgba(0, 255, 255, 0.5)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          transition: 'all 0.3s ease',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          '&:hover': {
            transform: 'scale(1.05)',
            boxShadow: '0 0 20px rgba(0, 255, 255, 0.4)',
          },
        },
        filled: {
          background: 'linear-gradient(45deg, rgba(0, 255, 255, 0.2), rgba(0, 153, 255, 0.2))',
          color: '#00ffff',
          boxShadow: '0 0 15px rgba(0, 255, 255, 0.2)',
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          height: 8,
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          overflow: 'hidden',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent)',
            animation: 'shimmer 2s infinite',
          },
        },
        bar: {
          borderRadius: 10,
          background: 'linear-gradient(90deg, #00ffff 0%, #ff0080 50%, #00ff88 100%)',
          boxShadow: '0 0 20px rgba(0, 255, 255, 0.5)',
          '&::after': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)',
            animation: 'progressGlow 1.5s infinite',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(10px)',
            transition: 'all 0.3s ease',
            '& fieldset': {
              borderColor: 'rgba(0, 255, 255, 0.3)',
              borderWidth: '1px',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(0, 255, 255, 0.6)',
              boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#00ffff',
              borderWidth: '2px',
              boxShadow: '0 0 30px rgba(0, 255, 255, 0.4)',
            },
          },
          '& .MuiInputLabel-root': {
            color: 'rgba(255, 255, 255, 0.7)',
            '&.Mui-focused': {
              color: '#00ffff',
            },
          },
          '& .MuiOutlinedInput-input': {
            color: '#ffffff',
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          background: 'rgba(0, 0, 0, 0.6)',
          borderRadius: 12,
          padding: '4px',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
        },
        indicator: {
          background: 'linear-gradient(90deg, #00ffff 0%, #ff0080 100%)',
          height: 3,
          borderRadius: '3px 3px 0 0',
          boxShadow: '0 0 15px rgba(0, 255, 255, 0.5)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: '0 4px',
          transition: 'all 0.3s ease',
          textTransform: 'uppercase',
          fontWeight: 600,
          letterSpacing: '0.05em',
          '&:hover': {
            background: 'rgba(0, 255, 255, 0.1)',
            boxShadow: '0 0 15px rgba(0, 255, 255, 0.2)',
          },
          '&.Mui-selected': {
            background: 'rgba(0, 255, 255, 0.2)',
            color: '#00ffff',
            boxShadow: '0 0 20px rgba(0, 255, 255, 0.3)',
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
          background: 'rgba(0, 0, 0, 0.8)',
        },
      },
    },
  },
});

// Custom keyframes for animations
const globalStyles = `
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
  
  @keyframes progressGlow {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
  }
  
  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  @keyframes glow {
    0%, 100% { 
      text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
      box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    50% { 
      text-shadow: 0 0 30px rgba(0, 255, 255, 0.8);
      box-shadow: 0 0 40px rgba(0, 255, 255, 0.5);
    }
  }
`;

// Enhanced animation variants for framer-motion
export const darkAnimationVariants = {
  fadeInUp: {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] },
  },
  fadeInScale: {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] },
  },
  slideInLeft: {
    initial: { opacity: 0, x: -50 },
    animate: { opacity: 1, x: 0 },
    transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] },
  },
  slideInRight: {
    initial: { opacity: 0, x: 50 },
    animate: { opacity: 1, x: 0 },
    transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] },
  },
  staggerContainer: {
    animate: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  },
  glowPulse: {
    animate: {
      boxShadow: [
        '0 0 20px rgba(0, 255, 255, 0.3)',
        '0 0 40px rgba(0, 255, 255, 0.6)',
        '0 0 60px rgba(0, 255, 255, 0.4)',
        '0 0 20px rgba(0, 255, 255, 0.3)',
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut",
      },
    },
  },
  neonGlow: {
    animate: {
      textShadow: [
        '0 0 20px rgba(0, 255, 255, 0.5)',
        '0 0 30px rgba(0, 255, 255, 0.8)',
        '0 0 40px rgba(0, 255, 255, 0.6)',
        '0 0 20px rgba(0, 255, 255, 0.5)',
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut",
      },
    },
  },
};

// Premium dark custom styles
export const darkCustomStyles = {
  neonCard: {
    background: 'rgba(0, 0, 0, 0.8)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(0, 255, 255, 0.3)',
    borderRadius: '16px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6), 0 0 40px rgba(0, 255, 255, 0.2)',
    position: 'relative',
    '&::before': {
      content: '""',
      position: 'absolute',
      top: -1,
      left: -1,
      right: -1,
      bottom: -1,
      borderRadius: '16px',
      background: 'linear-gradient(45deg, #00ffff, #ff0080, #00ff88)',
      zIndex: -1,
      opacity: 0.3,
    },
  },
  neonText: {
    background: 'linear-gradient(45deg, #00ffff 0%, #ff0080 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textShadow: '0 0 20px rgba(0, 255, 255, 0.5)',
  },
  glowButton: {
    background: 'linear-gradient(45deg, #00ffff 0%, #0099ff 100%)',
    border: '1px solid rgba(0, 255, 255, 0.5)',
    borderRadius: '8px',
    boxShadow: '0 0 20px rgba(0, 255, 255, 0.4)',
    color: '#000000',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    transition: 'all 0.3s ease',
    '&:hover': {
      boxShadow: '0 0 30px rgba(0, 255, 255, 0.6)',
      transform: 'translateY(-2px) scale(1.05)',
    },
  },
  floatingAnimation: {
    animation: 'float 3s ease-in-out infinite',
  },
  pulseAnimation: {
    animation: 'pulse 2s ease-in-out infinite',
  },
  glowAnimation: {
    animation: 'glow 2s ease-in-out infinite',
  },
};

// Inject global styles
if (typeof document !== 'undefined') {
  const styleElement = document.createElement('style');
  styleElement.textContent = globalStyles;
  document.head.appendChild(styleElement);
}

export default premiumDarkTheme;
