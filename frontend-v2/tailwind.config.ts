import type { Config } from 'tailwindcss';
import colors from 'tailwindcss/colors';

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    colors: {
      ...colors,
      // frontend-v2 palette remap (keeps existing classnames)
      slate: colors.zinc,
      sky: colors.violet,
    },
    extend: {
      boxShadow: {
        soft: '0 10px 30px rgba(0,0,0,.35)',
      },
    },
  },
  plugins: [],
} satisfies Config;
