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


def start_training(
    base_url: str,
    *,
    api_key: str | None,
    epochs: int,
    seq_len: int,
    batch_size: int,
    train_long: bool,
    train_intraday: bool,
    lookback_days_long: int | None,
    lookback_days_intraday: int | None,
    horizon_steps_long: int | None,
    horizon_steps_intraday: int | None,
    min_samples: int,
    max_symbols: int,
    after: str | None,
    page_size: int,
) -> str:
    url = base_url.rstrip("/") + "/api/learning/train-nse-eq-deep-async"
    params = {
        "epochs": int(epochs),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "train_long": bool(train_long),
        "train_intraday": bool(train_intraday),
        "lookback_days_long": (None if lookback_days_long is None else int(lookback_days_long)),
        "lookback_days_intraday": (None if lookback_days_intraday is None else int(lookback_days_intraday)),
        "horizon_steps_long": (None if horizon_steps_long is None else int(horizon_steps_long)),
        "horizon_steps_intraday": (None if horizon_steps_intraday is None else int(horizon_steps_intraday)),
        "min_samples": int(min_samples),
        "max_symbols": int(max_symbols),
        "after": after,
        "page_size": int(page_size),
    }
    params = {k: v for k, v in params.items() if v is not None}
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

    p = argparse.ArgumentParser(description="Import NSE_EQ universe from Upstox and train deep models on GPU")
    sub = p.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import", help="Import Upstox NSE_EQ instruments into instrument_meta")
    imp.add_argument("--limit", type=int, default=0, help="For testing: limit number of imported symbols")

    run = sub.add_parser("run", help="Start GPU deep training over NSE_EQ universe")
    run.add_argument("--epochs", type=int, default=3)
    run.add_argument("--seq-len", type=int, default=120)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--train-long", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--train-intraday", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--lookback-days-long", type=int, default=None)
    run.add_argument("--lookback-days-intraday", type=int, default=None)
    run.add_argument("--horizon-steps-long", type=int, default=None)
    run.add_argument("--horizon-steps-intraday", type=int, default=None)
    run.add_argument("--min-samples", type=int, default=500)
    run.add_argument("--max-symbols", type=int, default=0, help="0 means all")
    run.add_argument("--after", default=None, help="Cursor: start after this instrument_key")
    run.add_argument("--page-size", type=int, default=500)
    run.add_argument("--poll-seconds", type=float, default=2.0)

    watch = sub.add_parser("watch", help="Watch a training job")
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--poll-seconds", type=float, default=2.0)

    args = p.parse_args(rest)
    args.base_url = str(pre_args.base_url)
    args.api_key = pre_args.api_key

    if args.cmd == "import":
        out = import_universe(args.base_url, api_key=args.api_key, limit=int(args.limit))
        print(out)
        return 0

    if args.cmd == "watch":
        return watch_job(args.base_url, api_key=args.api_key, job_id=str(args.job_id), poll_seconds=float(args.poll_seconds))

    # run
    try:
        job_id = start_training(
            args.base_url,
            api_key=args.api_key,
            epochs=int(args.epochs),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            train_long=bool(args.train_long),
            train_intraday=bool(args.train_intraday),
            lookback_days_long=(None if args.lookback_days_long is None else int(args.lookback_days_long)),
            lookback_days_intraday=(None if args.lookback_days_intraday is None else int(args.lookback_days_intraday)),
            horizon_steps_long=(None if args.horizon_steps_long is None else int(args.horizon_steps_long)),
            horizon_steps_intraday=(None if args.horizon_steps_intraday is None else int(args.horizon_steps_intraday)),
            min_samples=int(args.min_samples),
            max_symbols=int(args.max_symbols),
            after=args.after,
            page_size=int(args.page_size),
        )
    except Exception as e:
        print(str(e))
        return 2

    print(f"Started NSE_EQ deep training job: {job_id}")
    return watch_job(args.base_url, api_key=args.api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
