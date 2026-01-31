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


def _start_job(base_url: str, *, api_key: str | None, path: str, params: dict[str, Any] | None = None, json_body: dict[str, Any] | None = None) -> str:
    url = base_url.rstrip("/") + path
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=_headers(api_key), params=params or None, json=json_body)
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"start failed {path}: {data}")
        job_id = (data or {}).get("job_id")
        if not job_id:
            raise RuntimeError(f"no job_id returned from {path}: {data}")
        return str(job_id)


def _watch_job(base_url: str, *, api_key: str | None, job_id: str, poll_seconds: float) -> int:
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
            metrics = job.get("metrics") or {}

            eta_s = metrics.get("eta_seconds") if isinstance(metrics, dict) else None
            elapsed_s = metrics.get("elapsed_seconds") if isinstance(metrics, dict) else None

            def _fmt_secs(v: Any) -> str:
                try:
                    v = int(v)
                except Exception:
                    return "--"
                if v < 0:
                    return "--"
                h = v // 3600
                m = (v % 3600) // 60
                s = v % 60
                if h > 0:
                    return f"{h:d}h{m:02d}m"
                if m > 0:
                    return f"{m:d}m{s:02d}s"
                return f"{s:d}s"

            eta_txt = _fmt_secs(eta_s)
            elapsed_txt = _fmt_secs(elapsed_s)

            prefix = f"[{status:9s}] {progress*100:6.2f}%  elapsed={elapsed_txt}  eta={eta_txt}"
            if msg and msg != last_msg:
                last_msg = msg
                print(f"{prefix}  {msg}")
            elif not msg:
                print(prefix)

            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                if status != "SUCCEEDED":
                    err = job.get("error")
                    if err:
                        print(f"Job error: {err}")
                    return 1
                return 0

            time.sleep(max(0.2, float(poll_seconds)))


def _print_artifacts(base_url: str, *, api_key: str | None) -> None:
    url = base_url.rstrip("/") + "/api/learning/artifacts"
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, headers=_headers(api_key))
            data = _json_or_text(r)
            if r.status_code >= 400:
                print(f"Could not fetch artifacts: {data}")
                return
            artifacts = (data or {}).get("artifacts") or []
            print("Artifacts:")
            for a in artifacts:
                name = (a or {}).get("name")
                path = (a or {}).get("path")
                exists = (a or {}).get("exists")
                size = (a or {}).get("size_bytes")
                print(f"- {name}: {path}  exists={exists}  size_bytes={size}")
    except Exception as e:
        print(f"Could not fetch artifacts: {e}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Master trainer: trigger training for all available model families")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--poll-seconds", type=float, default=2.0)

    p.add_argument("--run-ridge-batch", action=argparse.BooleanOptionalAction, default=True, help="Train classic (ridge) models for DEFAULT_UNIVERSE")
    p.add_argument("--run-deep-nse-eq", action=argparse.BooleanOptionalAction, default=True, help="Train deep long+intraday for NSE_EQ universe (requires CUDA)")
    p.add_argument("--run-pattern-seq", action=argparse.BooleanOptionalAction, default=True, help="Train sequence-based candlestick pattern model (torch CPU ok)")

    # Deep NSE_EQ params (matches /api/learning/train-nse-eq-deep-async)
    p.add_argument("--deep-epochs", type=int, default=6)
    p.add_argument("--deep-seq-len", type=int, default=120)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--deep-max-symbols", type=int, default=0)

    # Pattern seq params (matches /api/learning/train-nse-eq-pattern-seq-async)
    p.add_argument("--ps-interval", default="1m")
    p.add_argument("--ps-lookback-days", type=int, default=30)
    p.add_argument("--ps-seq-len", type=int, default=64)
    p.add_argument("--ps-stride", type=int, default=2)
    p.add_argument("--ps-epochs", type=int, default=3)
    p.add_argument("--ps-batch-size", type=int, default=256)
    p.add_argument("--ps-lr", type=float, default=1e-3)
    p.add_argument("--ps-label-threshold", type=float, default=0.35)
    p.add_argument("--ps-max-symbols", type=int, default=0)

    args = p.parse_args(argv)

    base_url = str(args.base_url)
    api_key = args.api_key

    # 1) Ridge batch (fast, CPU)
    if bool(args.run_ridge_batch):
        print("Starting ridge batch training...")
        job_id = _start_job(base_url, api_key=api_key, path="/api/learning/train-batch-async", json_body={"use_presets": True})
        print(f"ridge batch job_id={job_id}")
        rc = _watch_job(base_url, api_key=api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))
        if rc != 0:
            return rc

    # 2) Deep NSE_EQ (GPU)
    if bool(args.run_deep_nse_eq):
        print("Starting NSE_EQ deep training...")
        params = {
            "epochs": int(args.deep_epochs),
            "seq_len": int(args.deep_seq_len),
            "batch_size": int(args.deep_batch_size),
            "max_symbols": int(args.deep_max_symbols),
        }
        job_id = _start_job(base_url, api_key=api_key, path="/api/learning/train-nse-eq-deep-async", params=params)
        print(f"deep nse_eq job_id={job_id}")
        rc = _watch_job(base_url, api_key=api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))
        if rc != 0:
            return rc

    # 3) Pattern seq (torch CPU)
    if bool(args.run_pattern_seq):
        print("Starting NSE_EQ pattern sequence model training...")
        params = {
            "interval": str(args.ps_interval),
            "lookback_days": int(args.ps_lookback_days),
            "seq_len": int(args.ps_seq_len),
            "stride": int(args.ps_stride),
            "epochs": int(args.ps_epochs),
            "batch_size": int(args.ps_batch_size),
            "lr": float(args.ps_lr),
            "label_threshold": float(args.ps_label_threshold),
            "max_symbols": int(args.ps_max_symbols),
        }
        job_id = _start_job(base_url, api_key=api_key, path="/api/learning/train-nse-eq-pattern-seq-async", params=params)
        print(f"pattern seq job_id={job_id}")
        rc = _watch_job(base_url, api_key=api_key, job_id=job_id, poll_seconds=float(args.poll_seconds))
        if rc != 0:
            return rc

    print("All requested trainings finished.")
    _print_artifacts(base_url, api_key=api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
