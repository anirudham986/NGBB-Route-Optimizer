/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: { DEFAULT: '#1E3A5F', light: '#2A4A73' },
        teal: { DEFAULT: '#0D7377', light: '#10908F', dark: '#0A5C5F' },
        coral: '#A63A3A',
        amber: '#F59E0B',
      },
      fontFamily: {
        display: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
