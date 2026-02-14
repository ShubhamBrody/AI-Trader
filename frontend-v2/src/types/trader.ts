export type TraderAction = 'BUY' | 'SELL' | 'HOLD';
export type TraderSide = 'buy' | 'sell';

export type TraderAi = {
  action: TraderAction;
  confidence: number;
  uncertainty: number;
  model?: string | null;
};

export type TraderIndicators = {
  ema20: number;
  ema50: number;
  ema200: number;
  sma20: number;
  rsi14: number;
  bollinger: { mid: number; upper: number; lower: number; std?: number; bandwidth?: number };
  macd: { macd: number; signal: number; hist: number };
  stochastic: { k: number; d: number };
};

export type TraderDecisionPlan = {
  side: TraderSide;
  entry: number;
  stop: number;
  target: number;
  confidence?: number | null;
  source?: string | null;
  reason?: string | null;
};

export type TraderDecision = {
  action: TraderAction;
  plan?: TraderDecisionPlan | null;
  risk_multiplier: number;
  risk_fraction_suggested: number;
  reasons: string[];
};

export type TraderOverlays = {
  levels?: any;
  candle_patterns?: Array<Record<string, any>>;
  trade?: Record<string, any> | null;
};

export type TraderDecisionResponse = {
  ok: boolean;
  instrument_key: string;
  interval: string;
  asof_ts?: number | null;
  n_candles: number;
  last_close: number;
  ai: TraderAi;
  indicators: TraderIndicators;
  overlays: TraderOverlays;
  decision: TraderDecision;
  trend_confluence?: Record<string, any> | null;
  raw?: Record<string, any>;
};
