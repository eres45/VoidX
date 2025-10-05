import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';

const SpaceBackground = ({ children }) => {
  const canvasRef = useRef(null);
  const starsRef = useRef([]);
  const shootingStarsRef = useRef([]);
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Star class
    class Star {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.opacity = Math.random() * 0.8 + 0.2;
        this.twinkleSpeed = Math.random() * 0.02 + 0.01;
        this.twinklePhase = Math.random() * Math.PI * 2;
      }

      update() {
        this.twinklePhase += this.twinkleSpeed;
        this.opacity = 0.3 + Math.sin(this.twinklePhase) * 0.5;
      }

      draw() {
        ctx.save();
        ctx.globalAlpha = Math.max(0.1, this.opacity);
        
        // Create a glowing effect
        const gradient = ctx.createRadialGradient(
          this.x, this.y, 0,
          this.x, this.y, this.size * 3
        );
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.5, '#e6f3ff');
        gradient.addColorStop(1, 'transparent');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Bright center
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
      }
    }

    // Shooting Star class
    class ShootingStar {
      constructor() {
        this.reset();
        this.life = Math.random() * 100;
      }

      reset() {
        // Start from random edge
        const side = Math.floor(Math.random() * 4);
        switch (side) {
          case 0: // Top
            this.x = Math.random() * canvas.width;
            this.y = -50;
            this.vx = (Math.random() - 0.5) * 4;
            this.vy = Math.random() * 3 + 2;
            break;
          case 1: // Right
            this.x = canvas.width + 50;
            this.y = Math.random() * canvas.height;
            this.vx = -Math.random() * 3 - 2;
            this.vy = (Math.random() - 0.5) * 4;
            break;
          case 2: // Bottom
            this.x = Math.random() * canvas.width;
            this.y = canvas.height + 50;
            this.vx = (Math.random() - 0.5) * 4;
            this.vy = -Math.random() * 3 - 2;
            break;
          case 3: // Left
          default: // Fallback to left side
            this.x = -50;
            this.y = Math.random() * canvas.height;
            this.vx = Math.random() * 3 + 2;
            this.vy = (Math.random() - 0.5) * 4;
            break;
        }
        
        this.life = 0;
        this.maxLife = Math.random() * 60 + 40;
        this.size = Math.random() * 2 + 1;
        this.trail = [];
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life++;

        // Add to trail
        this.trail.push({ x: this.x, y: this.y });
        if (this.trail.length > 20) {
          this.trail.shift();
        }

        // Reset if off screen or life expired
        if (
          this.life > this.maxLife ||
          this.x < -100 || this.x > canvas.width + 100 ||
          this.y < -100 || this.y > canvas.height + 100
        ) {
          this.reset();
        }
      }

      draw() {
        if (this.trail.length < 2) return;

        ctx.save();
        
        // Draw trail
        for (let i = 0; i < this.trail.length - 1; i++) {
          const alpha = (i / this.trail.length) * 0.8;
          const size = (i / this.trail.length) * this.size;
          
          ctx.globalAlpha = alpha;
          
          // Glow effect
          const gradient = ctx.createRadialGradient(
            this.trail[i].x, this.trail[i].y, 0,
            this.trail[i].x, this.trail[i].y, size * 4
          );
          gradient.addColorStop(0, '#ffffff');
          gradient.addColorStop(0.5, '#cccccc');
          gradient.addColorStop(1, 'transparent');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(this.trail[i].x, this.trail[i].y, size * 4, 0, Math.PI * 2);
          ctx.fill();
          
          // Bright core
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(this.trail[i].x, this.trail[i].y, size, 0, Math.PI * 2);
          ctx.fill();
        }
        
        ctx.restore();
      }
    }

    // Initialize stars
    const createStars = () => {
      starsRef.current = [];
      const numStars = Math.floor((canvas.width * canvas.height) / 8000);
      
      for (let i = 0; i < numStars; i++) {
        starsRef.current.push(new Star());
      }
    };

    // Initialize shooting stars
    const createShootingStars = () => {
      shootingStarsRef.current = [];
      for (let i = 0; i < 3; i++) {
        shootingStarsRef.current.push(new ShootingStar());
      }
    };

    createStars();
    createShootingStars();

    // Animation loop
    const animate = () => {
      // Clear canvas with deep space black
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add subtle nebula effect
      const nebula = ctx.createRadialGradient(
        canvas.width * 0.3, canvas.height * 0.2, 0,
        canvas.width * 0.3, canvas.height * 0.2, canvas.width * 0.8
      );
      nebula.addColorStop(0, 'rgba(0, 20, 40, 0.1)');
      nebula.addColorStop(0.5, 'rgba(0, 10, 30, 0.05)');
      nebula.addColorStop(1, 'transparent');
      
      ctx.fillStyle = nebula;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Another nebula
      const nebula2 = ctx.createRadialGradient(
        canvas.width * 0.7, canvas.height * 0.8, 0,
        canvas.width * 0.7, canvas.height * 0.8, canvas.width * 0.6
      );
      nebula2.addColorStop(0, 'rgba(20, 0, 40, 0.08)');
      nebula2.addColorStop(0.5, 'rgba(10, 0, 20, 0.04)');
      nebula2.addColorStop(1, 'transparent');
      
      ctx.fillStyle = nebula2;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update and draw stars
      starsRef.current.forEach(star => {
        star.update();
        star.draw();
      });

      // Update and draw shooting stars
      shootingStarsRef.current.forEach(shootingStar => {
        shootingStar.update();
        shootingStar.draw();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <Box
      sx={{
        position: 'relative',
        minHeight: '100vh',
        width: '100%',
      }}
    >
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: -1,
          pointerEvents: 'none',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: '#000000',
            pointerEvents: 'none',
          }}
        />
      </Box>
      {children}
    </Box>
  );
};

export default SpaceBackground;
