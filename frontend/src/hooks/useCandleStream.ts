import { useEffect, useMemo, useRef, useState } from 'react';
import { CandleUpdateEvent, wsUrl } from '@/lib/ws';

type StreamState = {
  connected: boolean;
  lastMessageAt: number | null;
  error: string | null;
};

export function useCandleStream(enabled: boolean) {
  const [state, setState] = useState<StreamState>({
    connected: false,
    lastMessageAt: null,
    error: null,
  });
  const [lastCandle, setLastCandle] = useState<CandleUpdateEvent | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<number | null>(null);

  const url = useMemo(() => wsUrl('/api/ws/candles'), []);

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
          if (parsed?.type === 'candle_update') {
            setLastCandle(parsed as CandleUpdateEvent);
          }
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
