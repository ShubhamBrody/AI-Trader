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


def start_job(base_url: str, *, api_key: str | None, payload: dict[str, Any]) -> str:
    url = base_url.rstrip("/") + "/api/bootstrap/nse-eq-full-deep-async"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=_headers(api_key), json=payload)
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"start failed: {data}")
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
                    ik = metrics.get("instrument_key")
                    ds = metrics.get("done_symbols")
                    tl = metrics.get("trained_long")
                    ti = metrics.get("trained_intraday")
                    extra = f"  (symbol={ik} done={ds} long={tl} intra={ti})"
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

    p = argparse.ArgumentParser(description="Full NSE_EQ pipeline: import universe + backfill + GPU deep training")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Start job then watch")
    run.add_argument("--skip-universe-import", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--import-limit", type=int, default=0)
    run.add_argument("--after", default=None)
    run.add_argument("--page-size", type=int, default=200)
    run.add_argument("--max-symbols", type=int, default=0)

    run.add_argument("--backfill-daily", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--backfill-intraday", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--lookback-days-daily", type=int, default=1460)
    run.add_argument("--lookback-days-intraday", type=int, default=95)
    run.add_argument("--resume-from-db", action=argparse.BooleanOptionalAction, default=True)

    run.add_argument("--train-long", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--train-intraday", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--seq-len", type=int, default=120)
    run.add_argument("--epochs-long", type=int, default=6)
    run.add_argument("--epochs-intraday", type=int, default=6)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--min-samples", type=int, default=500)

    run.add_argument("--sleep-seconds-per-chunk", type=float, default=0.0)
    run.add_argument("--sleep-seconds-per-symbol", type=float, default=0.0)
    run.add_argument("--poll-seconds", type=float, default=2.0)

    watch = sub.add_parser("watch", help="Watch an existing job")
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--poll-seconds", type=float, default=2.0)

    args = p.parse_args(rest)

    base_url = str(pre_args.base_url)
    api_key = pre_args.api_key

    if args.cmd == "watch":
        return watch_job(base_url, api_key=api_key, job_id=str(args.job_id), poll_seconds=float(args.poll_seconds))

    payload = {
        "skip_universe_import": bool(args.skip_universe_import),
        "import_limit": int(args.import_limit),
        "after": args.after,
        "page_size": int(args.page_size),
        "max_symbols": int(args.max_symbols),
        "backfill_daily": bool(args.backfill_daily),
        "backfill_intraday": bool(args.backfill_intraday),
        "lookback_days_daily": int(args.lookback_days_daily),
        "lookback_days_intraday": int(args.lookback_days_intraday),
        "resume_from_db": bool(args.resume_from_db),
        "train_long": bool(args.train_long),
        "train_intraday": bool(args.train_intraday),
        "seq_len": int(args.seq_len),
        "epochs_long": int(args.epochs_long),
        "epochs_intraday": int(args.epochs_intraday),
        "batch_size": int(args.batch_size),
        "min_samples": int(args.min_samples),
        "sleep_seconds_per_chunk": float(args.sleep_seconds_per_chunk),
        "sleep_seconds_per_symbol": float(args.sleep_seconds_per_symbol),
    }

    try:
        job_id = start_job(base_url, api_key=api_key, payload=payload)
    except Exception as e:
        print(str(e))
        return 2

    print(f"Started full NSE_EQ deep pipeline job: {job_id}")
    return watch_job(base_url, api_key=api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
