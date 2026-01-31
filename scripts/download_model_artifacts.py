from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx


def _headers(api_key: str | None) -> dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


def _json_or_text(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}


def _download(base_url: str, *, api_key: str | None, name: str, out_dir: str) -> str:
    base = base_url.rstrip("/")
    list_url = base + "/api/learning/artifacts"
    with httpx.Client(timeout=60.0) as client:
        r = client.get(list_url, headers=_headers(api_key))
        data = _json_or_text(r)
        if r.status_code >= 400:
            raise RuntimeError(f"failed to list artifacts: {data}")

        artifacts = (data or {}).get("artifacts") or []
        info = next((a for a in artifacts if (a or {}).get("name") == name), None)
        if not info:
            raise RuntimeError(f"artifact not present: {name}")

        dl_url = base + f"/api/learning/artifacts/{name}"
        rr = client.get(dl_url, headers=_headers(api_key))
        if rr.status_code >= 400:
            raise RuntimeError(f"download failed: {_json_or_text(rr)}")

        filename = rr.headers.get("content-disposition")
        # We also accept server-provided filename via Content-Disposition, but default safely.
        out_name = None
        if filename and "filename=" in filename:
            try:
                out_name = filename.split("filename=", 1)[1].strip().strip('"')
            except Exception:
                out_name = None
        if not out_name:
            out_name = os.path.basename(str((info or {}).get("path") or f"{name}.bin"))

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(rr.content)
        return out_path


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Download trained model artifacts from a cloud deployment")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--out-dir", default="./downloads")
    p.add_argument("--artifact", choices=["db", "pattern_seq", "all"], default="all")
    args = p.parse_args(argv)

    names = ["db", "pattern_seq"] if args.artifact == "all" else [str(args.artifact)]
    for name in names:
        out_path = _download(str(args.base_url), api_key=args.api_key, name=name, out_dir=str(args.out_dir))
        print(f"Downloaded {name} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
