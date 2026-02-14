import { configureStore } from '@reduxjs/toolkit';
import { selectionReducer } from './selectionSlice';

export const store = configureStore({
  reducer: {
    selection: selectionReducer,
  },
});

// Persist a tiny amount of state so selection survives refresh.
store.subscribe(() => {
  try {
    const s = store.getState();
    localStorage.setItem('aitrader.selection', JSON.stringify(s.selection));
  } catch {
    // ignore
  }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
