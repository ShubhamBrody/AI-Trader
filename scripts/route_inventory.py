from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}


@dataclass
class Route:
    method: str
    path: str
    file: str
    line: int


def _get_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _find_router_prefixes(module: ast.Module) -> dict[str, str]:
    prefixes: dict[str, str] = {}

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        # router = APIRouter(prefix="/foo")
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "router":
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "APIRouter":
                    prefix = ""
                    for kw in node.value.keywords:
                        if kw.arg == "prefix":
                            s = _get_str(kw.value)
                            if s is not None:
                                prefix = s
                    prefixes["router"] = prefix
    return prefixes


def _join_paths(prefix: str, path: str) -> str:
    if not prefix:
        return path
    if not path:
        return prefix
    if prefix.endswith("/") and path.startswith("/"):
        return prefix[:-1] + path
    if not prefix.endswith("/") and not path.startswith("/"):
        return prefix + "/" + path
    return prefix + path


def inventory(api_dir: Path) -> list[Route]:
    routes: list[Route] = []

    for py in sorted(api_dir.glob("*.py")):
        if py.name.startswith("__"):
            continue
        src = py.read_text(encoding="utf-8")
        try:
            mod = ast.parse(src)
        except SyntaxError:
            continue

        prefixes = _find_router_prefixes(mod)
        router_prefix = prefixes.get("router", "")

        for node in ast.walk(mod):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                # @router.get("/path")
                if not isinstance(dec, ast.Call):
                    continue
                if not isinstance(dec.func, ast.Attribute):
                    continue
                if not isinstance(dec.func.value, ast.Name) or dec.func.value.id != "router":
                    continue

                method = dec.func.attr
                if method not in HTTP_METHODS and method != "websocket":
                    continue

                path = ""
                if dec.args:
                    s = _get_str(dec.args[0])
                    if s is not None:
                        path = s
                full = _join_paths(router_prefix, path)
                routes.append(Route(method=method.upper(), path=full, file=str(py), line=getattr(dec, "lineno", 1)))

    # stable sort
    routes.sort(key=lambda r: (r.path, r.method, r.file, r.line))
    return routes


def main() -> None:
    root = Path(os.environ.get("WORKSPACE_ROOT", ".")).resolve()
    current = root / "app" / "api"
    bc = root / "BackendComplete_extracted" / "BackendComplete" / "app" / "api"

    out = {
        "current": [r.__dict__ for r in inventory(current)],
        "backendcomplete": [r.__dict__ for r in inventory(bc)],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
