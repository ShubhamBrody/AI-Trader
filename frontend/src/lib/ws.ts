export type CandleUpdateEvent = {
  type: 'candle_update';
  instrument_key: string;
  interval: string;
  candle: {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  };
};

export function wsUrl(path: string) {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}${path}`;
}
