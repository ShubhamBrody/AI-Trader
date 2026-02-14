import { useEffect, useMemo, useRef } from 'react';
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
  type LogicalRange,
  ColorType,
  CrosshairMode,
  createChart,
  createTextWatermark,
} from 'lightweight-charts';

export type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type CandleOverlays = {
  supports?: number[];
  resistances?: number[];
  entry?: number;
  stop_loss?: number;
  target?: number;
};

export type DraggableOverlayKey = 'stop_loss' | 'target';

export type TradeZoneOverlay = {
  enabled?: boolean;
  side?: 'buy' | 'sell' | 'neutral' | string;
};

function toUtcTimestampSeconds(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

function formatIstFromUtcSeconds(seconds: number): string {
  const dt = new Date(seconds * 1000);
  return new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata',
    year: '2-digit',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(dt);
}

export function CandlestickChart({
  candles,
  overlays,
  watermark,
  tradeZone,
  lineOverlays,
  draggable,
  onDragEnd,
}: {
  candles: Candle[];
  overlays?: CandleOverlays;
  watermark?: string;
  tradeZone?: TradeZoneOverlay;
  lineOverlays?: Array<{
    id: string;
    title?: string;
    color: string;
    data: Array<{ time: UTCTimestamp; value: number }>;
    lineWidth?: number;
    lineStyle?: number;
  }>;
  draggable?: Partial<Record<DraggableOverlayKey, boolean>>;
  onDragEnd?: (key: DraggableOverlayKey, price: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const tradeZoneLayerRef = useRef<HTMLDivElement | null>(null);
  const tpZoneRef = useRef<HTMLDivElement | null>(null);
  const slZoneRef = useRef<HTMLDivElement | null>(null);
  const tpLabelRef = useRef<HTMLDivElement | null>(null);
  const slLabelRef = useRef<HTMLDivElement | null>(null);
  const sideBadgeRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const watermarkPluginRef = useRef<{ detach: () => void } | null>(null);
  const didInitialFitRef = useRef(false);
  const lastLogicalRangeRef = useRef<LogicalRange | null>(null);
  const suppressRangeCaptureRef = useRef(false);
  const overlayLinesRef = useRef<IPriceLine[]>([]);
  const namedLinesRef = useRef<Partial<Record<'entry' | 'stop_loss' | 'target', IPriceLine>>>({});
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());

  const updateTradeZoneOverlay = () => {
    const layer = tradeZoneLayerRef.current;
    const tpZone = tpZoneRef.current;
    const slZone = slZoneRef.current;
    const tpLabel = tpLabelRef.current;
    const slLabel = slLabelRef.current;
    const sideBadge = sideBadgeRef.current;
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    const container = containerRef.current;
    if (!layer || !tpZone || !slZone || !tpLabel || !slLabel || !sideBadge || !chart || !candleSeries) return;

    const enabled = Boolean(tradeZone?.enabled);

    const entry = typeof overlays?.entry === 'number' ? Number(overlays.entry) : null;
    const stopLoss = typeof overlays?.stop_loss === 'number' ? Number(overlays.stop_loss) : null;
    const target = typeof overlays?.target === 'number' ? Number(overlays.target) : null;

    const side = String(tradeZone?.side ?? '').toLowerCase();
    const isBuyLike = stopLoss != null && entry != null && target != null && stopLoss < entry && entry < target;
    const isSellLike = stopLoss != null && entry != null && target != null && target < entry && entry < stopLoss;

    // Zone overlay supports BUY-style and SELL-style plans.
    // BUY: SL < Entry < Target
    // SELL: Target < Entry < SL
    const shouldShow = enabled && entry != null && stopLoss != null && target != null && (isBuyLike || isSellLike) && data.candleData.length > 0;
    if (!shouldShow) {
      layer.style.display = 'none';
      return;
    }

    const yEntry = candleSeries.priceToCoordinate(entry);
    const ySL = candleSeries.priceToCoordinate(stopLoss);
    const yTP = candleSeries.priceToCoordinate(target);
    if (yEntry == null || ySL == null || yTP == null) {
      layer.style.display = 'none';
      return;
    }

    // Important: `layer` can be display:none when we call this. Measuring it would return 0x0,
    // so measure the chart container which is always visible.
    const rect = (container ?? layer).getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      layer.style.display = 'none';
      return;
    }

    const lastTime = data.candleData[data.candleData.length - 1]?.time;
    const xStartRaw = lastTime != null ? (chart.timeScale() as any).timeToCoordinate(lastTime) : null;
    const xStart = Number.isFinite(xStartRaw) ? Number(xStartRaw) : Math.floor(width * 0.35);
    const left = Math.max(0, Math.min(width - 1, xStart));
    const zoneWidth = Math.max(0, width - left);

    const setRect = (el: HTMLDivElement, topPx: number, bottomPx: number) => {
      const top = Math.max(0, Math.min(topPx, bottomPx));
      const bottom = Math.min(height, Math.max(topPx, bottomPx));
      const h = Math.max(0, bottom - top);
      el.style.left = `${left}px`;
      el.style.width = `${zoneWidth}px`;
      el.style.top = `${top}px`;
      el.style.height = `${h}px`;
      el.style.display = h >= 2 ? 'block' : 'none';
    };

    // Profit zone and risk zone depending on direction.
    // BUY: profit is Entry->Target (up), risk is SL->Entry (down)
    // SELL: profit is Target->Entry (down), risk is Entry->SL (up)
    if (isSellLike || side === 'sell') {
      setRect(tpZone, yEntry, yTP);
      setRect(slZone, ySL, yEntry);
    } else {
      setRect(tpZone, yTP, yEntry);
      setRect(slZone, yEntry, ySL);
    }

    // Labels
    tpLabel.style.left = `${left}px`;
    tpLabel.style.width = `${zoneWidth}px`;
    tpLabel.style.top = `${Math.max(0, Math.min(yTP, yEntry) + 10)}px`;
    tpLabel.style.display = tpZone.style.display;

    slLabel.style.left = `${left}px`;
    slLabel.style.width = `${zoneWidth}px`;
    slLabel.style.top = `${Math.max(0, Math.max(yEntry, ySL) - 34)}px`;
    slLabel.style.display = slZone.style.display;

    sideBadge.style.left = `${Math.min(width - 56, left + 10)}px`;
    sideBadge.style.top = `${Math.max(0, yEntry - 18)}px`;
    sideBadge.style.display = 'block';

    layer.style.display = 'block';
  };

  const dragStateRef = useRef<{
    active: DraggableOverlayKey | null;
    isDragging: boolean;
  }>({ active: null, isDragging: false });

  const data = useMemo(() => {
    const sorted = (candles ?? [])
      .slice()
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    const candleData = sorted.map((c) => ({
      time: toUtcTimestampSeconds(c.timestamp),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    const volumeData = sorted.map((c) => ({
      time: toUtcTimestampSeconds(c.timestamp),
      value: c.volume ?? 0,
      color: c.close >= c.open ? 'rgba(16, 185, 129, 0.55)' : 'rgba(244, 63, 94, 0.55)',
    }));

    return { candleData, volumeData };
  }, [candles]);

  // Create chart once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: 'rgba(2, 6, 23, 0)' },
        textColor: 'rgba(226, 232, 240, 0.85)',
        fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
      },
      grid: {
        vertLines: { color: 'rgba(51, 65, 85, 0.35)' },
        horzLines: { color: 'rgba(51, 65, 85, 0.35)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: 'rgba(51, 65, 85, 0.55)',
      },
      timeScale: {
        borderColor: 'rgba(51, 65, 85, 0.55)',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number | string) => {
          // `time` can be UTCTimestamp (seconds) for time charts.
          if (typeof time === 'number') return formatIstFromUtcSeconds(time);
          return null;
        },
      },
      localization: {
        dateFormat: "dd MMM 'yy",
        timeFormatter: (time: number | string) => {
          if (typeof time === 'number') return formatIstFromUtcSeconds(time);
          return '';
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#f43f5e',
      borderUpColor: '#10b981',
      borderDownColor: '#f43f5e',
      wickUpColor: '#10b981',
      wickDownColor: '#f43f5e',
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceScaleId: 'volume',
      priceFormat: { type: 'volume' },
      color: 'rgba(56, 189, 248, 0.35)',
      base: 0,
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    // Track user viewport (logical range) so we can restore it after data refreshes.
    // This prevents the chart from drifting back to the right when new candles arrive.
    const timeScale = chart.timeScale();
    const onLogicalRange = (r: LogicalRange | null) => {
      if (suppressRangeCaptureRef.current) return;
      lastLogicalRangeRef.current = r;
      if (r) didInitialFitRef.current = true;

      // Keep trade-zone overlay aligned with viewport changes.
      try {
        updateTradeZoneOverlay();
      } catch {
        // ignore
      }
    };
    timeScale.subscribeVisibleLogicalRangeChange(onLogicalRange);

    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      chart.applyOptions({ width: Math.floor(rect.width), height: Math.floor(rect.height) });
      try {
        updateTradeZoneOverlay();
      } catch {
        // ignore
      }
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      try {
        timeScale.unsubscribeVisibleLogicalRangeChange(onLogicalRange);
      } catch {
        // ignore
      }
      try {
        watermarkPluginRef.current?.detach();
      } catch {
        // ignore
      }
      watermarkPluginRef.current = null;

      // Remove indicator series.
      try {
        for (const s of indicatorSeriesRef.current.values()) {
          try {
            chart.removeSeries(s as any);
          } catch {
            // ignore
          }
        }
      } catch {
        // ignore
      }
      indicatorSeriesRef.current.clear();

      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, []);

  // Update watermark without recreating the chart.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // Treat watermark changes as a dataset switch (instrument). Reset the viewport
    // bookkeeping so the new dataset can fit once.
    didInitialFitRef.current = false;
    lastLogicalRangeRef.current = null;

    try {
      watermarkPluginRef.current?.detach();
    } catch {
      // ignore
    }
    watermarkPluginRef.current = null;

    if (!watermark) return;

    try {
      watermarkPluginRef.current = createTextWatermark(chart.panes()[0], {
        visible: true,
        horzAlign: 'center',
        vertAlign: 'center',
        lines: [
          {
            text: watermark,
            color: 'rgba(148, 163, 184, 0.15)',
            fontSize: 24,
            fontStyle: 'normal',
            fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
          },
        ],
      });
    } catch {
      // ignore
    }
  }, [watermark]);

  // Update series data.
  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    if (!chart || !candleSeries || !volumeSeries) return;

    const timeScale = chart.timeScale();
    const desiredLogicalRange = lastLogicalRangeRef.current ?? timeScale.getVisibleLogicalRange();

    suppressRangeCaptureRef.current = true;

    try {
      candleSeries.setData(data.candleData);
      volumeSeries.setData(data.volumeData);

      // Important: never call fitContent() on every refresh, as it resets user zoom/pan.
      // Preserve viewport across refreshes by restoring the previously visible *logical* range.
      if (!data.candleData.length) return;

      if (desiredLogicalRange) {
        try {
          timeScale.setVisibleLogicalRange(desiredLogicalRange);
          didInitialFitRef.current = true;
          return;
        } catch {
          // If the previous range is no longer valid, fall back to initial fit.
        }
      }

      if (!didInitialFitRef.current) {
        timeScale.fitContent();
        didInitialFitRef.current = true;
      }
    } finally {
      suppressRangeCaptureRef.current = false;
      try {
        updateTradeZoneOverlay();
      } catch {
        // ignore
      }
    }
  }, [data]);

  // Update overlay price lines.
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    if (!candleSeries) return;

    // Clear previously created lines.
    for (const line of overlayLinesRef.current) {
      try {
        candleSeries.removePriceLine(line);
      } catch {
        // ignore
      }
    }
    overlayLinesRef.current = [];
    namedLinesRef.current = {};

    // Lightweight-charts doesn't expose a full "clear all price lines" API.
    // Recreate series would be overkill; instead, we add a small, bounded number of lines
    // and rely on component re-mount when overlay set changes substantially.
    // Practically: keep to a few lines like the existing UI.

    const supports = (overlays?.supports ?? []).slice(0, 3);
    const resistances = (overlays?.resistances ?? []).slice(0, 3);

    const createLine = (price: number, color: string, title?: string) =>
      candleSeries.createPriceLine({
        price,
        color,
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: title ?? '',
      });

    if (!candles?.length) return;

    supports.forEach((p, idx) => overlayLinesRef.current.push(createLine(p, 'rgba(56,189,248,0.85)', `S${idx + 1}`)));
    resistances.forEach((p, idx) => overlayLinesRef.current.push(createLine(p, 'rgba(251,146,60,0.85)', `R${idx + 1}`)));

    if (typeof overlays?.entry === 'number') {
      const l = createLine(overlays.entry, 'rgba(34,197,94,0.9)', 'Entry');
      overlayLinesRef.current.push(l);
      namedLinesRef.current.entry = l;
    }
    if (typeof overlays?.stop_loss === 'number') {
      const l = createLine(overlays.stop_loss, 'rgba(244,63,94,0.9)', 'SL');
      overlayLinesRef.current.push(l);
      namedLinesRef.current.stop_loss = l;
    }
    if (typeof overlays?.target === 'number') {
      const l = createLine(overlays.target, 'rgba(59,130,246,0.9)', 'Target');
      overlayLinesRef.current.push(l);
      namedLinesRef.current.target = l;
    }

    try {
      updateTradeZoneOverlay();
    } catch {
      // ignore
    }
  }, [candles, overlays]);

  // Update indicator line overlays (EMA/VWAP/Bands, etc).
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const desired = (lineOverlays ?? []).filter((l) => l && l.id && Array.isArray(l.data));
    const desiredIds = new Set(desired.map((l) => l.id));

    // Remove old series.
    for (const [id, series] of indicatorSeriesRef.current.entries()) {
      if (!desiredIds.has(id)) {
        try {
          chart.removeSeries(series as any);
        } catch {
          // ignore
        }
        indicatorSeriesRef.current.delete(id);
      }
    }

    // Upsert series.
    for (const line of desired) {
      const lw = Math.max(1, Math.min(4, Math.round(Number(line.lineWidth ?? 2)))) as 1 | 2 | 3 | 4;
      const ls = (Number(line.lineStyle ?? 0) as any);
      let series = indicatorSeriesRef.current.get(line.id);
      if (!series) {
        series = chart.addSeries(LineSeries, {
          color: line.color,
          lineWidth: lw,
          lineStyle: ls,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        indicatorSeriesRef.current.set(line.id, series);
      } else {
        try {
          series.applyOptions({
            color: line.color,
            lineWidth: lw,
            lineStyle: ls,
          });
        } catch {
          // ignore
        }
      }

      try {
        (series as any).setData(line.data);
      } catch {
        // ignore
      }
    }

    return () => {
      // If component unmounts, cleanup is handled in the create-chart effect.
    };
  }, [lineOverlays]);

  // Update trade-zone overlay when toggled.
  useEffect(() => {
    try {
      updateTradeZoneOverlay();
    } catch {
      // ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tradeZone?.enabled, tradeZone?.side, overlays?.entry, overlays?.stop_loss, overlays?.target, candles]);

  // Draggable TP/SL lines (drag to modify).
  useEffect(() => {
    const el = containerRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!el || !candleSeries) return;

    const canDragSL = Boolean(draggable?.stop_loss);
    const canDragTP = Boolean(draggable?.target);
    if (!canDragSL && !canDragTP) return;

    const pickLineAtY = (y: number): DraggableOverlayKey | null => {
      const thresholdPx = 8;
      const candidates: Array<{ key: DraggableOverlayKey; dy: number }> = [];

      const sl = overlays?.stop_loss;
      const tp = overlays?.target;

      if (canDragSL && typeof sl === 'number') {
        const cy = candleSeries.priceToCoordinate(sl);
        if (cy != null) candidates.push({ key: 'stop_loss', dy: Math.abs(cy - y) });
      }
      if (canDragTP && typeof tp === 'number') {
        const cy = candleSeries.priceToCoordinate(tp);
        if (cy != null) candidates.push({ key: 'target', dy: Math.abs(cy - y) });
      }

      candidates.sort((a, b) => a.dy - b.dy);
      if (candidates.length && candidates[0].dy <= thresholdPx) return candidates[0].key;
      return null;
    };

    const onMouseDown = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      const y = e.clientY - rect.top;
      const hit = pickLineAtY(y);
      if (!hit) return;
      dragStateRef.current.active = hit;
      dragStateRef.current.isDragging = true;
      e.preventDefault();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!dragStateRef.current.isDragging) return;
      const key = dragStateRef.current.active;
      if (!key) return;

      const rect = el.getBoundingClientRect();
      const y = e.clientY - rect.top;
      const price = candleSeries.coordinateToPrice(y);
      if (price == null || !Number.isFinite(price)) return;

      const line = key === 'stop_loss' ? namedLinesRef.current.stop_loss : namedLinesRef.current.target;
      if (!line) return;
      try {
        line.applyOptions({ price: Number(price) });
      } catch {
        // ignore
      }
    };

    const endDrag = (e: MouseEvent) => {
      if (!dragStateRef.current.isDragging) return;
      const key = dragStateRef.current.active;
      dragStateRef.current.isDragging = false;
      dragStateRef.current.active = null;
      if (!key) return;
      const rect = el.getBoundingClientRect();
      const y = e.clientY - rect.top;
      const price = candleSeries.coordinateToPrice(y);
      if (price == null || !Number.isFinite(price)) return;
      onDragEnd?.(key, Number(price));
    };

    el.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', endDrag);

    return () => {
      el.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', endDrag);
    };
  }, [draggable?.stop_loss, draggable?.target, overlays?.stop_loss, overlays?.target, onDragEnd]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="absolute inset-0 h-full w-full" />
      <div ref={tradeZoneLayerRef} className="pointer-events-none absolute inset-0" aria-hidden style={{ display: 'none' }}>
        <div ref={tpZoneRef} className="absolute border border-sky-400/30 bg-sky-400/10" style={{ display: 'none' }} />
        <div ref={slZoneRef} className="absolute border border-emerald-400/30 bg-emerald-400/10" style={{ display: 'none' }} />

        <div ref={tpLabelRef} className="absolute flex justify-center text-[28px] font-semibold tracking-tight text-sky-300/70" style={{ display: 'none' }}>
          Sell Zone
        </div>
        <div ref={slLabelRef} className="absolute flex justify-center text-[28px] font-semibold tracking-tight text-emerald-300/70" style={{ display: 'none' }}>
          Buy Zone
        </div>

        <div
          ref={sideBadgeRef}
          className="absolute rounded-md bg-emerald-500 px-2 py-1 text-xs font-semibold text-slate-950"
          style={{ display: 'none' }}
        >
          Buy
        </div>
      </div>
    </div>
  );
}
