from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    p = Path("route_inventory.json")
    d = json.loads(p.read_text(encoding="utf-16"))

    cur = {(r["method"], r["path"]) for r in d["current"]}
    bc = {(r["method"], r["path"]) for r in d["backendcomplete"]}

    missing = sorted(bc - cur)
    extra = sorted(cur - bc)

    print(f"Missing vs BackendComplete: {len(missing)}")
    for m, path in missing:
        print(f"{m} {path}")

    print("\n---\n")
    print(f"Extra vs BackendComplete: {len(extra)}")
    for m, path in extra[:200]:
        print(f"{m} {path}")


if __name__ == "__main__":
    main()
