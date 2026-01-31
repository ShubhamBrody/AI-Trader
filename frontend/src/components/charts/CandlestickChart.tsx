import { useEffect, useMemo, useRef } from 'react';
import {
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
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
}: {
  candles: Candle[];
  overlays?: CandleOverlays;
  watermark?: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const overlayLinesRef = useRef<IPriceLine[]>([]);

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

    const watermarkPlugin = watermark
      ? createTextWatermark(chart.panes()[0], {
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
        })
      : null;

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      chart.applyOptions({ width: Math.floor(rect.width), height: Math.floor(rect.height) });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      watermarkPlugin?.detach();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, [watermark]);

  // Update series data.
  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    if (!chart || !candleSeries || !volumeSeries) return;

    candleSeries.setData(data.candleData);
    volumeSeries.setData(data.volumeData);

    chart.timeScale().fitContent();
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

    if (typeof overlays?.entry === 'number') overlayLinesRef.current.push(createLine(overlays.entry, 'rgba(34,197,94,0.9)', 'Entry'));
    if (typeof overlays?.stop_loss === 'number') overlayLinesRef.current.push(createLine(overlays.stop_loss, 'rgba(244,63,94,0.9)', 'SL'));
    if (typeof overlays?.target === 'number') overlayLinesRef.current.push(createLine(overlays.target, 'rgba(59,130,246,0.9)', 'Target'));
  }, [candles, overlays]);

  return <div ref={containerRef} className="h-full w-full" />;
}
