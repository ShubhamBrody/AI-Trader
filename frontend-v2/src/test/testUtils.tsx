import React, { type PropsWithChildren } from 'react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { render, type RenderOptions } from '@testing-library/react';
import { selectionReducer, type SelectionState } from '@/store/selectionSlice';

type RootState = {
  selection: SelectionState;
};

export function makeTestStore(preloadedSelection?: Partial<SelectionState>) {
  const preloadedState: RootState | undefined = preloadedSelection
    ? {
        selection: {
          instrument: {
            input: 'NSE_EQ|INE002A01018',
            ...preloadedSelection.instrument,
          },
        },
      }
    : undefined;

  return configureStore({
    reducer: { selection: selectionReducer } as any,
    preloadedState,
  });
}

export function renderWithProviders(
  ui: React.ReactElement,
  {
    route = '/',
    preloadedSelection,
    queryClient,
    store,
    ...renderOptions
  }: RenderOptions & {
    route?: string;
    preloadedSelection?: Partial<SelectionState>;
    queryClient?: QueryClient;
    store?: ReturnType<typeof makeTestStore>;
  } = {}
) {
  const qc = queryClient ??
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          refetchOnWindowFocus: false,
        },
      },
    });

  const st = store ?? makeTestStore(preloadedSelection);

  function Wrapper({ children }: PropsWithChildren) {
    return (
      <Provider store={st}>
        <QueryClientProvider client={qc}>
          <MemoryRouter initialEntries={[route]}>{children}</MemoryRouter>
        </QueryClientProvider>
      </Provider>
    );
  }

  return {
    store: st,
    queryClient: qc,
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
  };
}
