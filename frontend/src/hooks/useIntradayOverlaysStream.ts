import { useEffect, useMemo, useRef, useState } from 'react';
import { wsUrl } from '@/lib/ws';

export type IntradayOverlayAnalysis = {
  instrument_key: string;
  interval: string;
  asof_ts: number | null;
  n: number;
  atr?: number;
  // Backward-compatible fields used by older UI components.
  candle_patterns?: any[];
  winner_strategy?: any;
  trend?: {
    dir: 'up' | 'down' | 'flat';
    slope?: number;
    slope_pct?: number;
    strength?: number;
    window?: number;
  };
  levels?: {
    support: number[];
    resistance: number[];
  };
  patterns?: Array<{ type: string; side?: string; level?: number; confidence?: number }>;
  trade?: {
    side: 'buy' | 'sell';
    entry: number;
    stop: number;
    target: number;
    pattern_hint?: string;
    confidence?: number;
    model_confidence?: number;
    risk_profile?: 'conservative' | 'normal' | 'aggressive';
    reason?: string;
  } | null;
  ai?: {
    enabled: boolean;
    p_up?: number;
    model?: {
      instrument_key: string;
      interval: string;
      created_ts: number;
      metrics?: any;
      horizon_steps?: number;
      k_atr?: number;
    } | null;
    features?: Record<string, number>;
  };
};

export type IntradayOverlaysEvent = {
  type: 'intraday_overlays';
  instrument_key: string;
  interval: string;
  analysis: IntradayOverlayAnalysis;
  server_ts: number;
};

type StreamState = {
  connected: boolean;
  lastMessageAt: number | null;
  error: string | null;
};

export function useIntradayOverlaysStream(opts: {
  enabled: boolean;
  instrument_key: string | null | undefined;
  interval: string;
  lookback_minutes: number;
}) {
  const { enabled, instrument_key, interval, lookback_minutes } = opts;

  const [state, setState] = useState<StreamState>({
    connected: false,
    lastMessageAt: null,
    error: null,
  });
  const [last, setLast] = useState<IntradayOverlaysEvent | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<number | null>(null);

  const url = useMemo(() => {
    const base = wsUrl('/api/ws/intraday-overlays');
    const ik = instrument_key ? encodeURIComponent(instrument_key) : '';
    const q = new URLSearchParams();
    if (ik) q.set('instrument_key', instrument_key!);
    q.set('interval', interval);
    q.set('lookback_minutes', String(lookback_minutes));
    return `${base}?${q.toString()}`;
  }, [instrument_key, interval, lookback_minutes]);

  useEffect(() => {
    if (!enabled) return;
    if (!instrument_key) return;

    let cancelled = false;

    const connect = () => {
      if (cancelled) return;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setState({ connected: true, lastMessageAt: Date.now(), error: null });
        if (pingRef.current) window.clearInterval(pingRef.current);
        pingRef.current = window.setInterval(() => {
          try {
            ws.send('ping');
          } catch {
            // ignore
          }
        }, 20_000);
      };

      ws.onmessage = (evt) => {
        setState((s) => ({ ...s, lastMessageAt: Date.now() }));
        try {
          const parsed = JSON.parse(evt.data as string);
          if (parsed?.type === 'intraday_overlays') {
            setLast(parsed as IntradayOverlaysEvent);
          }
        } catch {
          // ignore
        }
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, error: 'WebSocket error' }));
      };

      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (pingRef.current) {
          window.clearInterval(pingRef.current);
          pingRef.current = null;
        }
        setTimeout(connect, 2000);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (pingRef.current) {
        window.clearInterval(pingRef.current);
        pingRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled, instrument_key, url]);

  return { state, last };
}
