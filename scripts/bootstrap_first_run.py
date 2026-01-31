from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx


def _json_or_text(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def start_bootstrap(
    base_url: str,
    *,
    api_key: str | None,
    instrument_keys: list[str] | None,
    intervals: list[str],
    lookback_days_daily: int,
    lookback_days_intraday: int,
    resume_from_db: bool,
    train_min_samples: int,
    train_lookback_days_long: int | None,
    train_lookback_days_intraday: int | None,
    cap_tier: str | None,
) -> str:
    payload: dict[str, Any] = {
        "instrument_keys": instrument_keys,
        "cap_tier": cap_tier,
        "intervals": intervals,
        "lookback_days_daily": lookback_days_daily,
        "lookback_days_intraday": lookback_days_intraday,
        "resume_from_db": bool(resume_from_db),
        "train_min_samples": int(train_min_samples),
        "train_lookback_days_long": train_lookback_days_long,
        "train_lookback_days_intraday": train_lookback_days_intraday,
    }

    url = base_url.rstrip("/") + "/api/bootstrap/first-run-async"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=_headers(api_key), json=payload)
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"bootstrap start failed: {data}")
        job_id = (data or {}).get("job_id")
        if not job_id:
            raise RuntimeError(f"bootstrap start returned no job_id: {data}")
        return str(job_id)


def watch_job(base_url: str, *, api_key: str | None, job_id: str, poll_seconds: float, tail: bool) -> int:
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

            if tail and msg and msg != last_msg:
                last_msg = msg
                print(f"[{status:9s}] {progress*100:6.2f}%  {msg}")
            elif not tail:
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
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default="http://127.0.0.1:8000")
    common.add_argument("--api-key", default=None)

    p = argparse.ArgumentParser(description="Run and/or watch the first-run bootstrap job", parents=[common])
    sub = p.add_subparsers(dest="cmd", required=False)

    run = sub.add_parser("run", help="Start bootstrap job then watch", parents=[common])
    run.add_argument("--instrument", action="append", dest="instruments", default=None, help="Repeatable instrument_key")
    run.add_argument("--cap-tier", default=None, choices=["large", "mid", "small", "unknown"], help="Use DB universe filtered by tier")
    run.add_argument("--interval", action="append", dest="intervals", default=["1d", "1m"], help="Repeatable interval")
    run.add_argument("--lookback-daily", type=int, default=365)
    run.add_argument("--lookback-intraday", type=int, default=30)
    run.add_argument("--resume-from-db", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--train-min-samples", type=int, default=200)
    run.add_argument("--train-lookback-long", type=int, default=None)
    run.add_argument("--train-lookback-intraday", type=int, default=None)
    run.add_argument("--poll-seconds", type=float, default=2.0)

    watch = sub.add_parser("watch", help="Watch an existing bootstrap/training job", parents=[common])
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--poll-seconds", type=float, default=2.0)

    args = p.parse_args(argv)

    cmd = args.cmd or "run"
    if cmd == "watch":
        return watch_job(args.base_url, api_key=args.api_key, job_id=args.job_id, poll_seconds=args.poll_seconds, tail=True)

    instruments = args.instruments
    if instruments is not None:
        instruments = [str(x).strip() for x in instruments if str(x).strip()]

    try:
        job_id = start_bootstrap(
            args.base_url,
            api_key=args.api_key,
            instrument_keys=instruments,
            cap_tier=args.cap_tier,
            intervals=[str(i).strip() for i in (args.intervals or []) if str(i).strip()],
            lookback_days_daily=int(args.lookback_daily),
            lookback_days_intraday=int(args.lookback_intraday),
            resume_from_db=bool(args.resume_from_db),
            train_min_samples=int(args.train_min_samples),
            train_lookback_days_long=args.train_lookback_long,
            train_lookback_days_intraday=args.train_lookback_intraday,
        )
    except Exception as e:
        print(str(e))
        return 2

    print(f"Started bootstrap job: {job_id}")
    return watch_job(args.base_url, api_key=args.api_key, job_id=job_id, poll_seconds=float(args.poll_seconds), tail=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
