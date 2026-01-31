from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx


def _headers(api_key: str | None) -> dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


def _json_or_text(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}


def import_universe(base_url: str, *, api_key: str | None, limit: int = 0) -> dict:
    url = base_url.rstrip("/") + "/api/universe/import-upstox-nse-eq"
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=_headers(api_key), params={"limit": int(limit)})
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"universe import failed: {data}")
        return data


def start_training(base_url: str, *, api_key: str | None, params: dict[str, Any]) -> str:
    url = base_url.rstrip("/") + "/api/learning/train-nse-eq-pattern-seq-async"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=_headers(api_key), params=params)
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"train start failed: {data}")
        job_id = (data or {}).get("job_id")
        if not job_id:
            raise RuntimeError(f"no job_id returned: {data}")
        return str(job_id)


def watch_job(base_url: str, *, api_key: str | None, job_id: str, poll_seconds: float) -> int:
    url = base_url.rstrip("/") + f"/api/learning/jobs/{job_id}"
    last_msg = None
    with httpx.Client(timeout=30.0) as client:
        while True:
            r = client.get(url, headers=_headers(api_key))
            data = _json_or_text(r)
            if r.status_code >= 400:
                print(f"ERROR: {data}")
                return 2

            job = (data or {}).get("job") or {}
            status = str(job.get("status") or "").upper()
            progress = float(job.get("progress") or 0.0)
            msg = job.get("message")
            metrics = job.get("metrics")

            if msg and msg != last_msg:
                last_msg = msg
                extra = ""
                if isinstance(metrics, dict):
                    sym = metrics.get("instrument_key") or metrics.get("done_symbols")
                    samples = metrics.get("samples") or metrics.get("total_windows")
                    loss = metrics.get("loss")
                    parts = []
                    if sym is not None:
                        parts.append(f"sym={sym}")
                    if samples is not None:
                        parts.append(f"samples={samples}")
                    if loss is not None:
                        parts.append(f"loss={loss}")
                    if parts:
                        extra = "  (" + " ".join(parts) + ")"
                print(f"[{status:9s}] {progress*100:6.2f}%  {msg}{extra}")
            elif not msg:
                print(f"[{status:9s}] {progress*100:6.2f}%")

            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                if status != "SUCCEEDED":
                    err = job.get("error")
                    if err:
                        print(f"Job error: {err}")
                    return 1
                return 0

            time.sleep(max(0.2, float(poll_seconds)))


def main(argv: list[str]) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--base-url", default="http://127.0.0.1:8000")
    pre.add_argument("--api-key", default=None)
    pre_args, rest = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(description="Train the sequence-based candlestick pattern model over NSE_EQ")
    sub = p.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import", help="Import Upstox NSE_EQ instruments into instrument_meta")
    imp.add_argument("--limit", type=int, default=0, help="For testing: limit number of imported symbols")

    run = sub.add_parser("run", help="Start training job then watch")
    run.add_argument("--interval", default="1m")
    run.add_argument("--lookback-days", type=int, default=30)
    run.add_argument("--seq-len", type=int, default=64)
    run.add_argument("--stride", type=int, default=2)
    run.add_argument("--epochs", type=int, default=3)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--lr", type=float, default=1e-3)
    run.add_argument("--label-threshold", type=float, default=0.35)

    run.add_argument("--max-candles-per-symbol", type=int, default=5000)
    run.add_argument("--max-windows-per-symbol", type=int, default=1500)
    run.add_argument("--max-windows-total", type=int, default=50000)

    run.add_argument("--max-symbols", type=int, default=0, help="0 means all")
    run.add_argument("--after", default=None)
    run.add_argument("--page-size", type=int, default=500)
    run.add_argument("--out-path", default=None, help="Override model output path")
    run.add_argument("--poll-seconds", type=float, default=2.0)

    watch = sub.add_parser("watch", help="Watch an existing job")
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--poll-seconds", type=float, default=2.0)

    args = p.parse_args(rest)
    base_url = str(pre_args.base_url)
    api_key = pre_args.api_key

    if args.cmd == "import":
        out = import_universe(base_url, api_key=api_key, limit=int(args.limit))
        print(out)
        return 0

    if args.cmd == "watch":
        return watch_job(base_url, api_key=api_key, job_id=str(args.job_id), poll_seconds=float(args.poll_seconds))

    params = {
        "interval": str(args.interval),
        "lookback_days": int(args.lookback_days),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "label_threshold": float(args.label_threshold),
        "max_candles_per_symbol": int(args.max_candles_per_symbol),
        "max_windows_per_symbol": int(args.max_windows_per_symbol),
        "max_windows_total": int(args.max_windows_total),
        "max_symbols": int(args.max_symbols),
        "after": args.after,
        "page_size": int(args.page_size),
        "out_path": args.out_path,
    }
    params = {k: v for k, v in params.items() if v is not None}

    try:
        job_id = start_training(base_url, api_key=api_key, params=params)
    except Exception as e:
        print(str(e))
        return 2

    print(f"Started NSE_EQ pattern-seq training job: {job_id}")
    return watch_job(base_url, api_key=api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
