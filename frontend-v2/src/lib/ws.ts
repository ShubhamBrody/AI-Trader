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
    // AI-first optional fields
    forecast_return?: number;
    predicted_close?: number;
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

    // AI-first price forecast used by overlays even when ML calibrator is disabled/unavailable.
    price?: {
      model?: string;
      model_family?: string | null;
      horizon_steps?: number | null;
      forecast_return?: number;
      predicted_close?: number;
      signal?: 'buy' | 'sell' | 'neutral' | string;
      confidence?: number;
      uncertainty?: number;
      metrics?: any;
    };
  };
};

export type IntradayOverlaysEvent = {
  type: 'intraday_overlays';
  instrument_key: string;
  interval: string;
  analysis: IntradayOverlayAnalysis;
  server_ts: number;
};

// Stable WS message envelope (matches EventBus: id/channel/type/payload/ts).
export type WsEnvelope<TPayload = any> = {
  id?: number | null;
  channel: string;
  type: string;
  payload: TPayload;
  ts: number;
};

export function parseWsEnvelope<TPayload = any>(msg: any): WsEnvelope<TPayload> | null {
  if (!msg || typeof msg !== 'object') return null;
  if (typeof msg.channel !== 'string') return null;
  if (typeof msg.type !== 'string') return null;
  if (typeof msg.ts !== 'number') return null;
  if (typeof msg.payload !== 'object') return null;
  return msg as WsEnvelope<TPayload>;
}

// --------------------
// EventBus channel payloads
// --------------------

// prediction
export type PredictionPredictorStartPayload = { instruments: string[]; cfg: Record<string, any> };
export type PredictionPredictorErrorPayload = { error: string };
export type PredictionCreatedPayload = {
  id: number;
  instrument_key: string;
  interval: string;
  horizon_steps: number;
  ts_pred: number;
  ts_target: number;
  pred_ret: number;
  pred_ret_adj: number;
  calib_bias: number;
  model_kind: string;
};
export type PredictionResolvedPayload = {
  id: number;
  instrument_key: string;
  interval: string;
  horizon_steps: number;
  ts_pred: number;
  ts_target: number;
  actual_ret: number;
  pred_ret_adj: number;
  error: number;
  calibration: Record<string, any>;
};
export type PredictionPredictorCyclePayload = { created: number; resolved: number; anchor: number };

// automation
export type AutomationRunStartedPayload = { job_id: string; kind: string; params: Record<string, any> };
export type AutomationRunUpdatedPayload = {
  job_id: string;
  progress?: number;
  message?: string;
  status?: string;
  stats?: Record<string, any>;
  error?: string;
};

// training
export type TrainingStartedPayload = { job_id: string; kind: string; instrument_key: string; interval: string };
export type TrainingProgressPayload = { job_id: string; progress: number; message: string; metrics: Record<string, any> };
export type TrainingSucceededPayload = { job_id: string; result: Record<string, any> };
export type TrainingFailedPayload = { job_id: string; result?: Record<string, any>; error?: string };

// hft
export type HftStartPayload = { broker: string; safe_mode: boolean; live_trading_enabled: boolean };
export type HftBrokerPayload = { broker: string };
export type HftErrorPayload = { error: string };
export type HftFlattenPayload = { closed: number; errors: string[] };
export type HftEntryPayload = { trade_id: number; instrument_key: string; qty: number; entry: number };
export type HftExitPayload = { trade_id: number; instrument_key: string; price: number; pnl: number; reason: string };
export type HftTradeRiskPayload = { trade_id: number; stop: number; target: number };
export type HftManageErrorPayload = { error: string };

// Candle stream payload (wrapped as WsEnvelope<CandleUpdatePayload> on the wire).
export type CandleUpdatePayload = {
  type: 'candle_update';
  instrument_key: string;
  interval: string;
  candle: CandleUpdateEvent['candle'];
  server_ts: number;
};

// Backward-compatible normalizer.
export function normalizeCandleUpdateEvent(msg: any): CandleUpdateEvent | null {
  if (!msg) return null;
  if (msg.type === 'candle_update' && msg.candle) return msg as CandleUpdateEvent;
  if (msg.type === 'candle_update' && msg.payload?.candle) {
    const p = msg.payload as CandleUpdatePayload;
    return {
      type: 'candle_update',
      instrument_key: String(p.instrument_key),
      interval: String(p.interval),
      candle: p.candle,
    };
  }
  return null;
}

// Backward-compatible normalizer.
export function normalizeIntradayOverlaysEvent(msg: any): IntradayOverlaysEvent | null {
  if (!msg) return null;
  if (msg.type === 'intraday_overlays' && msg.analysis) return msg as IntradayOverlaysEvent;
  if (msg.type === 'intraday_overlays' && msg.payload?.analysis) {
    const p = msg.payload as Partial<IntradayOverlaysEvent>;
    return {
      type: 'intraday_overlays',
      instrument_key: String(p.instrument_key ?? ''),
      interval: String(p.interval ?? ''),
      analysis: (p.analysis ?? {}) as IntradayOverlayAnalysis,
      server_ts: Number(p.server_ts ?? msg.ts ?? Date.now() / 1000),
    };
  }
  return null;
}

export function wsUrl(path: string) {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}${path}`;
}
