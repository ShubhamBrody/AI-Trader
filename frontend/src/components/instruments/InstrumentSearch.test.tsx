import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import { act } from 'react';

import { InstrumentSearch } from './InstrumentSearch';

const useInstrumentSuggestMock = vi.fn();

vi.mock('@/hooks/useInstrumentSuggest', () => {
  return {
    useInstrumentSuggest: (...args: any[]) => useInstrumentSuggestMock(...args),
  };
});

describe('InstrumentSearch', () => {
  it('debounces suggest calls for ~500ms idle', async () => {
    vi.useFakeTimers();

    useInstrumentSuggestMock.mockImplementation((query: string) => {
      return {
        data: { results: [] },
        isLoading: false,
        isError: false,
        query,
      };
    });

    function Harness() {
      const [v, setV] = React.useState('');
      return <InstrumentSearch value={v} onChange={setV} />;
    }

    render(<Harness />);

    const input = screen.getByPlaceholderText(/type a symbol/i);

    act(() => {
      fireEvent.change(input, { target: { value: 'R' } });
      fireEvent.change(input, { target: { value: 'RE' } });
      fireEvent.change(input, { target: { value: 'REL' } });
    });

    // Before 500ms, debounced value should still be empty (or prior value).
    vi.advanceTimersByTime(400);

    const queriesBefore = useInstrumentSuggestMock.mock.calls.map((c) => String(c[0] ?? ''));
    expect(queriesBefore.includes('REL')).toBe(false);

    // After 500ms idle, it should query with the latest value.
    await act(async () => {
      vi.advanceTimersByTime(200);
      await vi.runOnlyPendingTimersAsync();
    });

    const queriesAfter = useInstrumentSuggestMock.mock.calls.map((c) => String(c[0] ?? ''));
    expect(queriesAfter.includes('REL')).toBe(true);

    vi.useRealTimers();
  });
});
