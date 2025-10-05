import { createTheme } from '@mui/material/styles';

// Clean Black & White Theme - Professional & Minimalist
export const cleanDarkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#ffffff', // Pure white
      light: '#ffffff',
      dark: '#e0e0e0',
      contrastText: '#000000',
    },
    secondary: {
      main: '#cccccc', // Light gray
      light: '#e0e0e0',
      dark: '#999999',
      contrastText: '#000000',
    },
    background: {
      default: '#000000', // Pure black background
      paper: 'rgba(20, 20, 20, 0.95)', // Very dark gray with slight transparency
    },
    surface: {
      main: 'rgba(40, 40, 40, 0.8)',
      light: 'rgba(60, 60, 60, 0.9)',
      dark: 'rgba(20, 20, 20, 0.95)',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.9)',
      disabled: 'rgba(255, 255, 255, 0.4)',
    },
    error: {
      main: '#ffffff',
      light: '#e0e0e0',
      dark: '#cccccc',
    },
    warning: {
      main: '#ffffff',
      light: '#e0e0e0',
      dark: '#cccccc',
    },
    success: {
      main: '#ffffff',
      light: '#e0e0e0',
      dark: '#cccccc',
    },
    info: {
      main: '#ffffff',
      light: '#e0e0e0',
      dark: '#cccccc',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '3.5rem',
      fontWeight: 900,
      letterSpacing: '-0.02em',
      color: '#ffffff',
    },
    h2: {
      fontSize: '2.75rem',
      fontWeight: 800,
      letterSpacing: '-0.01em',
      color: '#ffffff',
    },
    h3: {
      fontSize: '2.25rem',
      fontWeight: 700,
      letterSpacing: '-0.01em',
      color: '#ffffff',
    },
    h4: {
      fontSize: '1.75rem',
      fontWeight: 600,
      color: '#ffffff',
    },
    h5: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#ffffff',
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
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
    },
  },
  shape: {
    borderRadius: 8,
  },
  shadows: [
    'none',
    '0 2px 8px rgba(0, 0, 0, 0.3)',
    '0 4px 16px rgba(0, 0, 0, 0.4)',
    '0 8px 24px rgba(0, 0, 0, 0.5)',
    '0 12px 32px rgba(0, 0, 0, 0.6)',
    '0 16px 40px rgba(0, 0, 0, 0.7)',
    '0 8px 32px rgba(0, 0, 0, 0.5)',
    '0 12px 40px rgba(0, 0, 0, 0.6)',
    '0 16px 48px rgba(0, 0, 0, 0.7)',
    '0 20px 60px rgba(0, 0, 0, 0.8)',
    '0 24px 80px rgba(0, 0, 0, 0.9)',
    '0 28px 100px rgba(0, 0, 0, 0.95)',
    '0 32px 120px rgba(0, 0, 0, 0.95)',
    '0 36px 140px rgba(0, 0, 0, 0.95)',
    '0 40px 160px rgba(0, 0, 0, 0.95)',
    '0 44px 180px rgba(0, 0, 0, 0.95)',
    '0 48px 200px rgba(0, 0, 0, 0.95)',
    '0 52px 220px rgba(0, 0, 0, 0.95)',
    '0 56px 240px rgba(0, 0, 0, 0.95)',
    '0 60px 260px rgba(0, 0, 0, 0.95)',
    '0 64px 280px rgba(0, 0, 0, 0.95)',
    '0 68px 300px rgba(0, 0, 0, 0.95)',
    '0 72px 320px rgba(0, 0, 0, 0.95)',
    '0 76px 340px rgba(0, 0, 0, 0.95)',
    '0 80px 360px rgba(0, 0, 0, 0.95)',
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: '#000000',
          backgroundAttachment: 'fixed',
          minHeight: '100vh',
          overflow: 'auto',
        },
        '*': {
          scrollbarWidth: 'thin',
          scrollbarColor: '#ffffff #333333',
        },
        '*::-webkit-scrollbar': {
          width: '8px',
        },
        '*::-webkit-scrollbar-track': {
          background: '#333333',
        },
        '*::-webkit-scrollbar-thumb': {
          background: '#ffffff',
          borderRadius: '4px',
          '&:hover': {
            background: '#e0e0e0',
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
          fontWeight: 600,
          padding: '12px 32px',
          letterSpacing: '0.05em',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-2px)',
          },
          '&:active': {
            transform: 'translateY(0)',
          },
        },
        contained: {
          background: '#ffffff',
          color: '#000000',
          boxShadow: '0 4px 20px rgba(255, 255, 255, 0.2)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          '&:hover': {
            background: '#e0e0e0',
            boxShadow: '0 6px 30px rgba(255, 255, 255, 0.3)',
          },
        },
        outlined: {
          borderColor: '#ffffff',
          color: '#ffffff',
          background: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(10px)',
          '&:hover': {
            borderColor: '#e0e0e0',
            background: 'rgba(255, 255, 255, 0.1)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          background: 'rgba(40, 40, 40, 0.8)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 16px 48px rgba(0, 0, 0, 0.8)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'rgba(40, 40, 40, 0.9)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
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
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.8)',
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 16,
          background: 'rgba(20, 20, 20, 0.98)',
          backdropFilter: 'blur(30px)',
          border: '2px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.9)',
        },
      },
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          color: '#ffffff',
          fontSize: '1.75rem',
          fontWeight: 700,
          padding: '24px 24px 16px',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          transition: 'all 0.3s ease',
          fontWeight: 500,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          fontSize: '0.75rem',
          '&:hover': {
            transform: 'scale(1.05)',
          },
        },
        filled: {
          background: 'rgba(255, 255, 255, 0.1)',
          color: '#ffffff',
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          height: 6,
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          overflow: 'hidden',
        },
        bar: {
          borderRadius: 4,
          background: '#ffffff',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(10px)',
            transition: 'all 0.3s ease',
            '& fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.2)',
              borderWidth: '1px',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.4)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#ffffff',
              borderWidth: '2px',
            },
          },
          '& .MuiInputLabel-root': {
            color: 'rgba(255, 255, 255, 0.7)',
            '&.Mui-focused': {
              color: '#ffffff',
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
          borderRadius: 8,
          padding: '4px',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
        indicator: {
          background: '#ffffff',
          height: 3,
          borderRadius: '3px 3px 0 0',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          margin: '0 4px',
          transition: 'all 0.3s ease',
          textTransform: 'uppercase',
          fontWeight: 500,
          letterSpacing: '0.05em',
          fontSize: '0.875rem',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.1)',
          },
          '&.Mui-selected': {
            background: 'rgba(255, 255, 255, 0.15)',
            color: '#ffffff',
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          background: 'rgba(0, 0, 0, 0.8)',
        },
      },
    },
  },
});

// Clean animation variants for framer-motion
export const cleanAnimationVariants = {
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
};

// Clean custom styles
export const cleanCustomStyles = {
  card: {
    background: 'rgba(40, 40, 40, 0.8)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '12px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
  },
  text: {
    color: '#ffffff',
    fontWeight: 600,
  },
  button: {
    background: '#ffffff',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '8px',
    color: '#000000',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    transition: 'all 0.3s ease',
    '&:hover': {
      background: '#e0e0e0',
      transform: 'translateY(-2px)',
    },
  },
  container: {
    background: 'rgba(0, 0, 0, 0.6)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '16px',
  },
};

export default cleanDarkTheme;
