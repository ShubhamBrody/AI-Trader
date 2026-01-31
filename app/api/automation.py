from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.automation.scheduler import AUTORETRAIN
from app.automation.runs import get_run, list_runs

router = APIRouter(prefix="/automation", tags=["automation"])


@router.get("/status")
def status() -> dict:
	return {"ok": True, **AUTORETRAIN.status()}


@router.post("/start")
async def start() -> dict:
	res = await AUTORETRAIN.start()
	if not res.get("ok"):
		raise HTTPException(status_code=400, detail=res)
	return res


@router.post("/stop")
async def stop() -> dict:
	return await AUTORETRAIN.stop()


@router.post("/run-once")
async def run_once(force: bool = False) -> dict:
	# Returns immediately; the actual orchestration runs as a tracked training job.
	res = await AUTORETRAIN.run_once(force=bool(force))
	if not res.get("ok"):
		raise HTTPException(status_code=400, detail=res)
	return res


@router.get("/runs")
def runs(limit: int = 50, status: str | None = None, kind: str | None = None) -> dict:
	return {"ok": True, "items": list_runs(limit=int(limit), status=status, kind=kind)}


@router.get("/runs/{job_id}")
def run_detail(job_id: str) -> dict:
	row = get_run(str(job_id))
	if not row:
		raise HTTPException(status_code=404, detail="not found")
	return {"ok": True, "run": row}
