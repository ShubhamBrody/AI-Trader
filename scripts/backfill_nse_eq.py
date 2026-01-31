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


def start_backfill(
    base_url: str,
    *,
    api_key: str | None,
    intervals: list[str],
    lookback_days_daily: int,
    lookback_days_intraday: int,
    resume_from_db: bool,
    page_size: int,
    after: str | None,
    max_symbols: int,
) -> str:
    url = base_url.rstrip("/") + "/api/data/backfill-nse-eq-async"
    body = {
        "intervals": [str(x) for x in intervals],
        "lookback_days_daily": int(lookback_days_daily),
        "lookback_days_intraday": int(lookback_days_intraday),
        "resume_from_db": bool(resume_from_db),
        "page_size": int(page_size),
        "after": after,
        "max_symbols": int(max_symbols),
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=_headers(api_key), json=body)
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"backfill start failed: {data}")
        job_id = (data or {}).get("job_id")
        if not job_id:
            raise RuntimeError(f"no job_id returned: {data}")
        return str(job_id)


def watch_job(base_url: str, *, api_key: str | None, job_id: str, poll_seconds: float) -> int:
    url = base_url.rstrip("/") + f"/api/data/jobs/{job_id}"
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

            if msg and msg != last_msg:
                last_msg = msg
                print(f"[{status:9s}] {progress*100:6.2f}%  {msg}")
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

    p = argparse.ArgumentParser(description="Backfill candles for all NSE_EQ equities (paged) with progress")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Start NSE_EQ backfill job")
    run.add_argument("--interval", action="append", dest="intervals", default=None, help="Repeatable. Example: --interval 1d --interval 1m")
    run.add_argument("--lookback-days-daily", type=int, default=365)
    run.add_argument("--lookback-days-intraday", type=int, default=5)
    run.add_argument("--resume-from-db", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--page-size", type=int, default=500)
    run.add_argument("--after", default=None)
    run.add_argument("--max-symbols", type=int, default=0, help="0 means all")
    run.add_argument("--poll-seconds", type=float, default=2.0)

    watch = sub.add_parser("watch", help="Watch an existing NSE_EQ backfill job")
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--poll-seconds", type=float, default=2.0)

    args = p.parse_args(rest)
    args.base_url = str(pre_args.base_url)
    args.api_key = pre_args.api_key

    if args.cmd == "watch":
        return watch_job(args.base_url, api_key=args.api_key, job_id=str(args.job_id), poll_seconds=float(args.poll_seconds))

    # run
    intervals = args.intervals or ["1d", "1m"]
    try:
        job_id = start_backfill(
            args.base_url,
            api_key=args.api_key,
            intervals=intervals,
            lookback_days_daily=int(args.lookback_days_daily),
            lookback_days_intraday=int(args.lookback_days_intraday),
            resume_from_db=bool(args.resume_from_db),
            page_size=int(args.page_size),
            after=args.after,
            max_symbols=int(args.max_symbols),
        )
    except Exception as e:
        print(str(e))
        return 2

    print(f"Started NSE_EQ backfill job: {job_id}")
    return watch_job(args.base_url, api_key=args.api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
