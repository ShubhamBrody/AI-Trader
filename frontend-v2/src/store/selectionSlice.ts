import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type SelectedInstrument = {
  input: string;
  instrument_key?: string;
  canonical_symbol?: string;
  tradingsymbol?: string;
  name?: string | null;
  updatedAt?: number;
};

export type SelectionState = {
  instrument: SelectedInstrument;
};

const DEFAULT_INPUT = 'NSE_EQ|INE002A01018';

function loadInitial(): SelectionState {
  try {
    const raw = localStorage.getItem('aitrader.selection');
    if (!raw) return { instrument: { input: DEFAULT_INPUT } };
    const parsed = JSON.parse(raw);
    const inst = parsed?.instrument;
    if (!inst || typeof inst.input !== 'string' || inst.input.length === 0) {
      return { instrument: { input: DEFAULT_INPUT } };
    }
    return {
      instrument: {
        input: inst.input,
        instrument_key: typeof inst.instrument_key === 'string' ? inst.instrument_key : undefined,
        canonical_symbol: typeof inst.canonical_symbol === 'string' ? inst.canonical_symbol : undefined,
        tradingsymbol: typeof inst.tradingsymbol === 'string' ? inst.tradingsymbol : undefined,
        name: typeof inst.name === 'string' || inst.name === null ? inst.name : undefined,
        updatedAt: typeof inst.updatedAt === 'number' ? inst.updatedAt : undefined,
      },
    };
  } catch {
    return { instrument: { input: DEFAULT_INPUT } };
  }
}

const initialState: SelectionState = loadInitial();

const selectionSlice = createSlice({
  name: 'selection',
  initialState,
  reducers: {
    setInstrumentInput(state, action: PayloadAction<string>) {
      state.instrument = {
        input: action.payload,
        updatedAt: Date.now(),
      };
    },
    setInstrumentPick(
      state,
      action: PayloadAction<{
        canonical_symbol: string;
        tradingsymbol: string;
        instrument_key: string;
        name?: string | null;
      }>
    ) {
      state.instrument = {
        input: action.payload.canonical_symbol,
        instrument_key: action.payload.instrument_key,
        canonical_symbol: action.payload.canonical_symbol,
        tradingsymbol: action.payload.tradingsymbol,
        name: action.payload.name ?? null,
        updatedAt: Date.now(),
      };
    },
  },
});

export const { setInstrumentInput, setInstrumentPick } = selectionSlice.actions;
export const selectionReducer = selectionSlice.reducer;
