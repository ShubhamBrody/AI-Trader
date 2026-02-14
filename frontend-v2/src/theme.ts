const THEME_KEY = 'aitrader-v2.theme';
const ACCENT_KEY = 'aitrader-v2.accent';
const TRANSITION_CLASS = 'theme-transition';
const TRANSITION_MS = 200;

type Theme = 'dark' | 'light';
export type Accent = 'sky' | 'violet' | 'emerald' | 'rose' | 'amber';

const ACCENT_CLASSES: Record<Accent, string> = {
  sky: 'accent-sky',
  violet: 'accent-violet',
  emerald: 'accent-emerald',
  rose: 'accent-rose',
  amber: 'accent-amber',
};

export function getTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // ignore
  }
  return 'dark';
}

export function getAccent(): Accent {
  try {
    const stored = localStorage.getItem(ACCENT_KEY);
    if (stored === 'sky' || stored === 'violet' || stored === 'emerald' || stored === 'rose' || stored === 'amber') return stored;
  } catch {
    // ignore
  }
  return 'sky';
}

export function setTheme(theme: Theme) {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // ignore
  }
  applyTheme(theme, { withTransition: true });
}

export function setAccent(accent: Accent) {
  try {
    localStorage.setItem(ACCENT_KEY, accent);
  } catch {
    // ignore
  }
  applyAccent(accent, { withTransition: true });
}

export function toggleTheme() {
  setTheme(getTheme() === 'dark' ? 'light' : 'dark');
}

export function initTheme() {
  // No transitions on first paint.
  applyTheme(getTheme(), { withTransition: false });
  applyAccent(getAccent(), { withTransition: false });
}

function applyTheme(theme: Theme, opts: { withTransition: boolean }) {
  const root = document.documentElement;

  if (opts.withTransition) {
    // Smooth toggle: enable transitions briefly around the class flip.
    try {
      root.classList.add(TRANSITION_CLASS);
      window.setTimeout(() => root.classList.remove(TRANSITION_CLASS), TRANSITION_MS);
    } catch {
      // ignore
    }
  }

  root.classList.toggle('dark', theme === 'dark');
}

function applyAccent(accent: Accent, opts: { withTransition: boolean }) {
  const root = document.documentElement;

  if (opts.withTransition) {
    try {
      root.classList.add(TRANSITION_CLASS);
      window.setTimeout(() => root.classList.remove(TRANSITION_CLASS), TRANSITION_MS);
    } catch {
      // ignore
    }
  }

  for (const cls of Object.values(ACCENT_CLASSES)) {
    root.classList.remove(cls);
  }
  root.classList.add(ACCENT_CLASSES[accent]);
}
