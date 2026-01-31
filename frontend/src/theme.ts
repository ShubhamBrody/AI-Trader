const THEME_KEY = 'aitrader.theme';

type Theme = 'dark' | 'light';

export function getTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return 'dark';
}

export function setTheme(theme: Theme) {
  localStorage.setItem(THEME_KEY, theme);
  applyTheme(theme);
}

export function toggleTheme() {
  setTheme(getTheme() === 'dark' ? 'light' : 'dark');
}

export function initTheme() {
  applyTheme(getTheme());
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.classList.toggle('dark', theme === 'dark');
  // Tailwind default light background is better with explicit class.
  document.body.classList.toggle('bg-slate-950', theme === 'dark');
  document.body.classList.toggle('text-slate-100', theme === 'dark');
  document.body.classList.toggle('bg-slate-50', theme === 'light');
  document.body.classList.toggle('text-slate-900', theme === 'light');
}
