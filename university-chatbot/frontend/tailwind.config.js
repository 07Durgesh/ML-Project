/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'primary-dark': '#1A202C', // Very dark blue/black
        'secondary-dark': '#2D3748', // Slightly lighter dark
        'accent-green': '#34D399',  // Brighter green for contrast
        'text-light': '#E2E8F0',    // Light text color
        'text-muted': '#A0AEC0',    // Muted text for descriptions
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: 0, transform: 'translateY(20px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}