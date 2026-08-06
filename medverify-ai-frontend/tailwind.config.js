/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        ink: {
          950: '#0A0F16',
          900: '#0E1520',
          800: '#141D2B',
          700: '#1C2836',
          600: '#2A3A4D',
          500: '#3F5468',
        },
        mist: {
          50: '#F7F9FB',
          100: '#EEF2F6',
          200: '#DEE6ED',
          300: '#C3D0DC',
          400: '#94A7B8',
        },
        clinical: {
          50: '#EFF9FB',
          100: '#D9F1F5',
          400: '#3FB8CF',
          500: '#1B9CB5',
          600: '#0F7E96',
          700: '#0D6478',
        },
        trust: {
          500: '#2F6FED',
          600: '#2358C4',
          700: '#1B449A',
        },
        verdict: {
          support: '#1E9E6B',
          warn: '#C88A1E',
          contradict: '#C4453B',
          neutral: '#7C8CA0',
        },
      },
      boxShadow: {
        card: '0 1px 2px rgba(10,15,22,0.04), 0 8px 24px -12px rgba(10,15,22,0.12)',
        'card-lg': '0 4px 12px rgba(10,15,22,0.06), 0 24px 48px -16px rgba(10,15,22,0.18)',
        glow: '0 0 0 1px rgba(63,184,207,0.25), 0 0 32px rgba(63,184,207,0.18)',
      },
      backgroundImage: {
        'grid-faint':
          'linear-gradient(to right, rgba(148,167,184,0.08) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,167,184,0.08) 1px, transparent 1px)',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-400px 0' },
          '100%': { backgroundPosition: '400px 0' },
        },
        pulseRing: {
          '0%': { boxShadow: '0 0 0 0 rgba(63,184,207,0.35)' },
          '100%': { boxShadow: '0 0 0 14px rgba(63,184,207,0)' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        shimmer: 'shimmer 1.6s linear infinite',
        pulseRing: 'pulseRing 1.8s ease-out infinite',
        fadeUp: 'fadeUp 0.5s ease-out both',
      },
    },
  },
  plugins: [],
}
