import '@testing-library/jest-dom/vitest';

// Some components/libraries assume ResizeObserver exists.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// @ts-expect-error - test env shim
globalThis.ResizeObserver = globalThis.ResizeObserver ?? ResizeObserverStub;

const originalWarn = console.warn;
console.warn = (...args: any[]) => {
  const msg = String(args?.[0] ?? '');
  if (msg.includes('React Router Future Flag Warning')) return;
  originalWarn(...args);
};
