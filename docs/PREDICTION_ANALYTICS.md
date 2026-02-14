# Prediction Analytics

This page documents the JSON shapes for the prediction analytics endpoints.

## Rollups

`GET /api/prediction/analytics/rollups?since_days=30&limit=200`

Optional filters:
- `instrument_key`
- `interval`
- `horizon_steps`

Example response:
```json
{
  "ok": true,
  "since_days": 30,
  "items": [
    {
      "instrument_key": "NSE_EQ|RELIANCE",
      "interval": "1m",
      "horizon_steps": 60,
      "n": 1240,
      "accuracy": 0.54,
      "mae": 0.0061,
      "rmse": 0.0089,
      "mean_error": -0.0003,
      "avg_pred": 0.0002,
      "avg_actual": 0.0005,
      "confusion": {"tp": 360, "tn": 310, "fp": 290, "fn": 280}
    }
  ]
}
```

## Reliability bins

`GET /api/prediction/analytics/reliability?since_days=30`

Buckets results by `abs(pred_ret_adj)`.

Example response:
```json
{
  "ok": true,
  "since_days": 30,
  "bins": [0, 0.001, 0.0025, 0.005, 0.01, 0.02, 1.0],
  "items": [
    {"lo": 0, "hi": 0.001, "n": 420, "accuracy": 0.51, "mae": 0.0031, "mean_error": 0.0001},
    {"lo": 0.001, "hi": 0.0025, "n": 310, "accuracy": 0.53, "mae": 0.0044, "mean_error": -0.0002}
  ]
}
```

## Drift

`GET /api/prediction/analytics/drift?baseline_days=30&recent_days=7`

Compares quality in the recent window against the baseline window (baseline excludes the recent period).

Example response:
```json
{
  "ok": true,
  "baseline_days": 30,
  "recent_days": 7,
  "items": [
    {
      "key": "NSE_EQ|RELIANCE::1m::h60",
      "baseline": {"instrument_key": "NSE_EQ|RELIANCE", "interval": "1m", "horizon_steps": 60, "n": 900, "accuracy": 0.55, "mae": 0.0060, "rmse": 0.0088, "mean_error": -0.0002, "avg_pred": 0.0001, "avg_actual": 0.0004, "confusion": {"tp": 270, "tn": 225, "fp": 210, "fn": 195}},
      "recent": {"instrument_key": "NSE_EQ|RELIANCE", "interval": "1m", "horizon_steps": 60, "n": 220, "accuracy": 0.50, "mae": 0.0072, "rmse": 0.0101, "mean_error": 0.0006, "avg_pred": -0.0004, "avg_actual": 0.0001, "confusion": {"tp": 70, "tn": 40, "fp": 60, "fn": 50}},
      "delta": {"accuracy": -0.05, "mae": 0.0012, "rmse": 0.0013, "mean_error": 0.0008}
    }
  ]
}
```

## Calibration snapshot list

`GET /api/prediction/analytics/calibrations?instrument_key=NSE_EQ|RELIANCE&interval=1m`

Returns the current EMA calibration rows from the `online_calibration` table.

Example response:
```json
{
  "ok": true,
  "items": [
    {"key": "NSE_EQ|RELIANCE::1m::h60", "updated_ts": 1739570000, "n": 1200, "bias": -0.0003, "mae": 0.0061}
  ]
}
```
