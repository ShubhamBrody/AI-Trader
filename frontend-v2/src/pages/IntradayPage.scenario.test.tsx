import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { screen } from '@testing-library/react';

import { renderWithProviders } from '@/test/testUtils';
import { IntradayPage } from './IntradayPage';

vi.mock('@/components/charts/CandlestickChart', () => {
  return {
    CandlestickChart: () => <div data-testid="candlestick-chart" />, // avoid lightweight-charts in tests
  };
});

vi.mock('@/components/instruments/InstrumentSearch', () => {
  return {
    InstrumentSearch: ({ value, onChange }: any) => (
      <input aria-label="Stock" value={value} onChange={(e) => onChange(e.target.value)} />
    ),
  };
});

vi.mock('@/hooks/useInstrumentResolve', () => {
  return {
    useInstrumentResolve: () => ({
      data: {
        instrument_key: 'NSE_EQ|TESTKEY',
        canonical_symbol: 'TEST',
        tradingsymbol: 'TEST',
        name: 'Test Co',
      },
      isError: false,
      isLoading: false,
    }),
  };
});

const useIntradayOverlaysStreamMock = vi.fn();
vi.mock('@/hooks/useIntradayOverlaysStream', () => {
  return {
    useIntradayOverlaysStream: (...args: any[]) => useIntradayOverlaysStreamMock(...args),
  };
});

const apiGetMock = vi.fn();
const apiPostMock = vi.fn();
vi.mock('@/lib/api', () => {
  return {
    apiGet: (...args: any[]) => apiGetMock(...args),
    apiPost: (...args: any[]) => apiPostMock(...args),
  };
});

describe('IntradayPage scenario', () => {
  it('renders winner strategy and candlestick pattern windows table', async () => {
    apiGetMock.mockResolvedValue([
      { timestamp: '2026-01-31T09:15:00Z', open: 100, high: 101, low: 99, close: 100.5, volume: 10 },
      { timestamp: '2026-01-31T09:16:00Z', open: 100.5, high: 102, low: 100, close: 101.5, volume: 12 },
    ]);
    apiPostMock.mockResolvedValue({ status: 'polled' });

    useIntradayOverlaysStreamMock.mockReturnValue({
      state: { connected: true, lastMessageAt: Date.now(), error: null },
      last: {
        analysis: {
          asof_ts: 1769831760,
          trend: { dir: 'up', strength: 0.6 },
          levels: { support: [99.0], resistance: [102.0] },
          candle_patterns: [
            {
              name: 'Hammer',
              family: 'bullish_reversal',
              side: 'buy',
              window: 1,
              start_ts: 1769831700,
              end_ts: 1769831760,
              confidence: 0.8,
              details: { weight: 1.1 },
            },
          ],
          winner_strategy: {
            id: 'breakout_above_resistance',
            label: 'breakout above resistance',
            side: 'buy',
            confidence: 0.78,
            source: 'trade',
            pattern_hint: { name: 'Hammer', confidence: 0.8 },
          },
          trade: {
            side: 'buy',
            entry: 101.5,
            stop: 100.0,
            target: 104.0,
            confidence: 0.78,
            reason: 'breakout_above_resistance',
            pattern_hint: { name: 'Hammer', confidence: 0.8 },
          },
          patterns: [],
          ai: { enabled: false },
        },
      },
    });

    renderWithProviders(<IntradayPage />);

    expect(await screen.findByText('Setups')).toBeInTheDocument();

    // Winner card
    expect(screen.getByText(/winner strategy/i)).toBeInTheDocument();
    expect(screen.getByText(/breakout above resistance/i)).toBeInTheDocument();
    expect(screen.getByText('78%')).toBeInTheDocument();
    expect(screen.getByText(/pattern hint/i)).toBeInTheDocument();
    expect(screen.getAllByText(/hammer/i).length).toBeGreaterThan(0);

    // Table
    expect(screen.getByText('Start')).toBeInTheDocument();
    expect(screen.getByText('End')).toBeInTheDocument();
    expect(screen.getByText('Type')).toBeInTheDocument();
    expect(screen.getByText('Side')).toBeInTheDocument();
    expect(screen.getByText('Conf')).toBeInTheDocument();

    expect(screen.getByText('Hammer')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();
  });

  it('renders empty states when no setup/patterns', async () => {
    apiGetMock.mockResolvedValue([]);
    apiPostMock.mockResolvedValue({ status: 'polled' });

    useIntradayOverlaysStreamMock.mockReturnValue({
      state: { connected: true, lastMessageAt: Date.now(), error: null },
      last: {
        analysis: {
          asof_ts: 1769831760,
          trend: { dir: 'flat', strength: 0.1 },
          levels: { support: [], resistance: [] },
          candle_patterns: [],
          patterns: [],
          trade: null,
          ai: { enabled: false },
        },
      },
    });

    renderWithProviders(<IntradayPage />);

    expect(await screen.findByText('Setups')).toBeInTheDocument();
    expect(screen.getByText(/no setup detected/i)).toBeInTheDocument();
    expect(screen.getByText(/no candlestick patterns detected/i)).toBeInTheDocument();
  });
});
