import { useEffect, useMemo, useRef, useState } from 'react';
import { CandleUpdateEvent, normalizeCandleUpdateEvent, wsUrl } from '@/lib/ws';

type StreamState = {
  connected: boolean;
  lastMessageAt: number | null;
  error: string | null;
};

type CandleStreamOptions = {
  instrument_key?: string;
  interval?: string;
  lookback_minutes?: number;
  poll_seconds?: number;
};

export function useCandleStream(enabled: boolean, opts?: CandleStreamOptions) {
  const [state, setState] = useState<StreamState>({
    connected: false,
    lastMessageAt: null,
    error: null,
  });
  const [lastCandle, setLastCandle] = useState<CandleUpdateEvent | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<number | null>(null);

  const url = useMemo(() => {
    const params = new URLSearchParams();
    if (opts?.instrument_key) params.set('instrument_key', opts.instrument_key);
    if (opts?.interval) params.set('interval', opts.interval);
    if (typeof opts?.lookback_minutes === 'number') params.set('lookback_minutes', String(opts.lookback_minutes));
    if (typeof opts?.poll_seconds === 'number') params.set('poll_seconds', String(opts.poll_seconds));
    const qs = params.toString();
    return wsUrl(`/api/ws/candles${qs ? `?${qs}` : ''}`);
  }, [opts?.instrument_key, opts?.interval, opts?.lookback_minutes, opts?.poll_seconds]);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    const connect = () => {
      if (cancelled) return;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setState({ connected: true, lastMessageAt: Date.now(), error: null });

        // backend keeps connection alive by receiving text
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
          const ev = normalizeCandleUpdateEvent(parsed);
          if (ev) setLastCandle(ev as CandleUpdateEvent);
        } catch {
          // ignore non-JSON
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
        // reconnect
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
  }, [enabled, url]);

  return { state, lastCandle };
}
