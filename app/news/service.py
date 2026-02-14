from __future__ import annotations

import html as _html
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import httpx

from app.core.db import db_conn
from app.core.settings import settings


_POS = {"surge", "gain", "beats", "record", "growth", "up", "rally", "strong", "bull", "profit"}
_NEG = {"fall", "drops", "miss", "loss", "down", "weak", "bear", "fraud", "probe", "crash"}


_NEWS_AI_CACHE_TTL_SECONDS = 120


_AI_ENGINE_SINGLETON = None


def _get_ai_engine():
    global _AI_ENGINE_SINGLETON
    if _AI_ENGINE_SINGLETON is None:
        from app.ai.engine import AIEngine

        _AI_ENGINE_SINGLETON = AIEngine()
    return _AI_ENGINE_SINGLETON


@lru_cache(maxsize=4096)
def _ai_predict_cached(instrument_key: str, interval: str, horizon_steps: int, bucket: int) -> dict:
    _ = bucket  # cache-busting TTL bucket
    eng = _get_ai_engine()
    return eng.predict(instrument_key, interval=interval, horizon_steps=int(horizon_steps))


def _safe_sentiment(title: str, summary: str | None) -> float:
    """Returns sentiment in [-1, 1] using a simple keyword heuristic.

    We keep this dependency-free so tests and offline dev keep working.
    """
    text = f"{title} {summary or ''}".lower()
    score = 0.0
    for w in _POS:
        if w in text:
            score += 0.15
    for w in _NEG:
        if w in text:
            score -= 0.18
    return float(max(-1.0, min(1.0, score)))


def _impact(sentiment: float, title: str) -> float:
    # Rough impact proxy: more extreme sentiment and more specific titles get higher impact.
    length = max(1, len(title))
    specificity = 1.0 - math.exp(-length / 80.0)
    return float(max(0.0, min(1.0, abs(sentiment) * 0.7 + specificity * 0.3)))


def _parse_rss_urls() -> list[str]:
    urls = [u.strip() for u in (settings.NEWS_RSS_URLS or "").split(",") if u.strip()]
    return urls


@dataclass(frozen=True)
class NewsItem:
    ts: int
    source: str
    title: str
    url: str
    summary: str | None
    sentiment: float
    impact: float


class NewsService:
    def _impact_label(self, impact: float) -> str:
        v = float(max(0.0, min(1.0, impact)))
        if v >= 0.66:
            return "HIGH"
        if v >= 0.33:
            return "MEDIUM"
        return "LOW"

    def _extract_text_from_html(self, html_text: str) -> str:
        # Very lightweight (dependency-free) extraction.
        # Remove scripts/styles, then strip tags.
        s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html_text or "")
        s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
        s = re.sub(r"(?is)<!--.*?-->", " ", s)
        s = re.sub(r"(?is)<[^>]+>", " ", s)
        s = _html.unescape(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _fetch_article_text(self, url: str) -> str | None:
        u = (url or "").strip()
        if not u or not (u.startswith("http://") or u.startswith("https://")):
            return None

        try:
            with httpx.Client(follow_redirects=True, timeout=6.0) as client:
                r = client.get(
                    u,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                )
        except Exception:
            return None

        if r.status_code != 200:
            return None
        ctype = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            return None

        text = self._extract_text_from_html(r.text)
        # Keep a reasonable chunk for the model.
        if len(text) < 200:
            return None
        return text[:9000]

    def _ollama_chat(self, messages: list[dict[str, str]]) -> str | None:
        base = (settings.OLLAMA_BASE_URL or "").strip().rstrip("/")
        if not base:
            return None

        payload: dict[str, Any] = {
            "model": settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 120,
            },
        }

        try:
            with httpx.Client(timeout=12.0) as client:
                r = client.post(f"{base}/api/chat", json=payload)
            if r.status_code != 200:
                return None
            data = r.json()
            content = (((data or {}).get("message") or {}).get("content") or "").strip()
            if not content:
                return None
            content = re.sub(r"\s+", " ", content).strip()
            # Hard guard against runaway output.
            return content[:500]
        except Exception:
            return None

    def _get_cached_llm_summary(self, url: str) -> str | None:
        u = (url or "").strip()
        if not u:
            return None
        ttl_days = max(1, int(settings.NEWS_LLM_SUMMARY_TTL_DAYS or 7))
        min_ts = int(time.time()) - ttl_days * 24 * 60 * 60
        with db_conn() as conn:
            row = conn.execute(
                "SELECT summary, created_ts FROM news_llm_summaries WHERE url=?",
                (u,),
            ).fetchone()
            if not row:
                return None
            created_ts = int(row[1] or 0)
            if created_ts < min_ts:
                return None
            s = (row[0] or "").strip()
            return s or None

    def _put_cached_llm_summary(self, *, url: str, summary: str, title: str | None, source: str | None) -> None:
        u = (url or "").strip()
        s = (summary or "").strip()
        if not u or not s:
            return
        created_ts = int(time.time())
        with db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO news_llm_summaries (url, summary, model, created_ts, title, source) VALUES (?, ?, ?, ?, ?, ?)",
                (u, s, settings.OLLAMA_MODEL, created_ts, (title or None), (source or None)),
            )

    def llm_summary_for(self, *, title: str, url: str | None, rss_summary: str | None, source: str | None = None) -> str | None:
        if not bool(settings.NEWS_LLM_SUMMARY_ENABLED):
            return None

        link = (url or "").strip()
        if not link:
            return None

        cached = self._get_cached_llm_summary(link)
        if cached is not None:
            return cached

        article_text = None
        try:
            article_text = self._fetch_article_text(link)
        except Exception:
            article_text = None

        # Still allow summarization from RSS snippet if the article fetch fails.
        snippet = (rss_summary or "").strip()
        if not article_text and not snippet:
            return None

        sys = (
            "You summarize business/market news for traders. "
            "Write a concise plain-text summary in 1-2 sentences (<=45 words). "
            "No bullet points, no hashtags, no advice."
        )
        user = (
            f"Title: {title.strip()}\n"
            f"Feed snippet: {snippet[:1200] if snippet else ''}\n"
            f"Article text (may be partial): {(article_text or '')[:7000]}\n\n"
            "Task: Summarize what this news is about in 1-2 sentences."
        )
        out = self._ollama_chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ])

        if out:
            try:
                self._put_cached_llm_summary(url=link, summary=out, title=title, source=source)
            except Exception:
                pass
            return out

        return None

    def enrich_item_for_ui(self, it: dict[str, Any]) -> dict[str, Any]:
        ts = int(it.get("ts") or 0)
        title = str(it.get("title") or "")
        url = (it.get("url") or "")
        source = it.get("source")
        impact = float(it.get("impact") or 0.0)

        iso = None
        try:
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            iso = None

        llm = None
        try:
            llm = self.llm_summary_for(title=title, url=url, rss_summary=(it.get("summary") or None), source=source)
        except Exception:
            llm = None

        return {
            **it,
            # Fields used by frontend cards
            "id": f"news:{url}" if url else f"news:{ts}:{abs(hash(title))}",
            "timestamp": iso,
            "published_at": iso,
            "region": None,
            "impact_score": impact,
            "impact_label": self._impact_label(impact),
            # Prefer LLM summary when available; fall back to RSS snippet.
            "summary_text": (llm or it.get("summary") or None),
        }

    def _tag_macro(self, title: str, summary: str | None) -> list[str]:
        text = f"{title} {summary or ''}".lower()
        tags: list[str] = []
        rules: list[tuple[str, list[str]]] = [
            ("rbi", ["rbi", "repo rate", "mpc"]),
            ("fed", ["fed", "fomc", "powell"]),
            ("inflation", ["inflation", "cpi", "wpi"]),
            ("gdp", ["gdp"]),
            ("usd-inr", ["usd/inr", "usdinr", "rupee", "inr"]),
            ("oil", ["crude", "brent", "wti", "oil prices"]),
            ("gold", ["gold", "xau"]),
            ("silver", ["silver", "xag"]),
        ]
        for tag, needles in rules:
            if any(n in text for n in needles):
                tags.append(tag)
        return tags

    def _ai_sentiment_from_mentions(self, mentions: list[dict[str, Any]], *, interval: str = "1d") -> tuple[float, float] | None:
        """Compute sentiment/impact using AI forecasts for mentioned instruments.

        Returns:
            (sentiment, impact) in [-1, 1] x [0, 1], or None if unavailable.
        """

        if not mentions:
            return None

        # Keep this conservative: score only a few most relevant mentions.
        ranked = sorted(mentions, key=lambda m: -float(m.get("relevance") or 0.0))
        ranked = [m for m in ranked if (m.get("instrument_key") or "").strip()][:4]
        if not ranked:
            return None

        bucket = int(time.time() // int(_NEWS_AI_CACHE_TTL_SECONDS))

        sent_sum = 0.0
        impact_sum = 0.0
        weight_sum = 0.0

        for m in ranked:
            ik = str(m.get("instrument_key") or "").strip()
            w = float(m.get("relevance") or 0.0)
            if not ik or w <= 0:
                continue

            try:
                pred = _ai_predict_cached(ik, interval, 1, bucket)
                feats = pred.get("features") or {}
                last_close = float(feats.get("last_close") or 0.0)
                next_close = float((((pred.get("prediction") or {}).get("next_hour_ohlc") or {}).get("close")) or 0.0)
                if last_close <= 0 or next_close <= 0:
                    continue

                ret = (next_close / last_close) - 1.0

                # Map return into [-1, 1] smoothly.
                s = float(max(-1.0, min(1.0, math.tanh(ret * 50.0))))

                p = pred.get("prediction") or {}
                conf = float(p.get("confidence") or 0.0)
                unc = float(p.get("uncertainty") or 1.0)
                strength = float(max(0.0, min(1.0, conf * (1.0 - unc))))

                # Impact favors agreement/strength and magnitude.
                imp = float(max(0.0, min(1.0, 0.6 * abs(s) + 0.4 * strength)))

                sent_sum += s * w
                impact_sum += imp * w
                weight_sum += w
            except Exception:
                continue

        if weight_sum <= 0:
            return None

        sentiment = float(sent_sum / weight_sum)
        impact = float(max(0.0, min(1.0, impact_sum / weight_sum)))
        return (sentiment, impact)

    @lru_cache(maxsize=1)
    def _instrument_lookup_cache(self) -> dict[str, Any]:
        """Build in-memory lookup maps for fast matching.

        Cached and invalidated manually by calling _instrument_lookup_cache.cache_clear().
        """
        with db_conn() as conn:
            meta_rows = conn.execute(
                "SELECT instrument_key, tradingsymbol, updated_ts FROM instrument_meta WHERE tradingsymbol IS NOT NULL AND tradingsymbol <> ''"
            ).fetchall()
            extra_rows = conn.execute(
                "SELECT instrument_key, name, underlying_symbol, exchange, segment, instrument_type, updated_ts FROM instrument_extra"
            ).fetchall()

        sym_to_keys: dict[str, list[str]] = {}
        name_tokens_to_keys: dict[str, list[str]] = {}
        key_to_info: dict[str, dict[str, Any]] = {}

        def _add(map_obj: dict[str, list[str]], k: str, instrument_key: str) -> None:
            if not k:
                return
            map_obj.setdefault(k, [])
            if instrument_key not in map_obj[k]:
                map_obj[k].append(instrument_key)

        max_ts = 0
        for r in meta_rows:
            ik = str(r[0])
            sym = str(r[1] or "").strip().upper()
            ts = int(r[2] or 0)
            max_ts = max(max_ts, ts)
            if sym:
                _add(sym_to_keys, sym, ik)
            key_to_info.setdefault(ik, {})
            key_to_info[ik].update({"instrument_key": ik, "tradingsymbol": sym})

        for r in extra_rows:
            ik = str(r[0])
            name = str(r[1] or "").strip()
            underlying = str(r[2] or "").strip().upper() or None
            exchange = str(r[3] or "").strip().upper() or None
            segment = str(r[4] or "").strip().upper() or None
            instrument_type = str(r[5] or "").strip().upper() or None
            ts = int(r[6] or 0)
            max_ts = max(max_ts, ts)

            key_to_info.setdefault(ik, {})
            key_to_info[ik].update(
                {
                    "instrument_key": ik,
                    "name": name or None,
                    "underlying_symbol": underlying,
                    "exchange": exchange,
                    "segment": segment,
                    "instrument_type": instrument_type,
                }
            )

            if underlying:
                _add(sym_to_keys, underlying, ik)
            # Tokenize name to improve matches (but keep it conservative)
            for tok in re.split(r"[^A-Za-z0-9]+", name.upper()):
                if len(tok) >= 4:
                    _add(name_tokens_to_keys, tok, ik)

        return {"sym_to_keys": sym_to_keys, "name_tokens_to_keys": name_tokens_to_keys, "key_to_info": key_to_info, "max_ts": max_ts}

    def _extract_mentions(self, title: str, summary: str | None, *, max_mentions: int = 8) -> list[dict[str, Any]]:
        """Best-effort entity extraction from title/summary.

        Output:
          [{instrument_key, relevance, reason, ...instrument info...}]
        """
        text = f"{title} {summary or ''}".strip()
        if not text:
            return []

        up = text.upper()
        # Simple candidate tokens: uppercase-ish words/numbers.
        tokens = [t for t in re.split(r"[^A-Z0-9]+", up) if t]
        tokens = [t for t in tokens if 2 <= len(t) <= 15]
        # Prefer longer tokens first
        tokens = sorted(set(tokens), key=lambda x: (-len(x), x))

        lookup = self._instrument_lookup_cache()
        sym_to_keys: dict[str, list[str]] = lookup["sym_to_keys"]
        name_tokens_to_keys: dict[str, list[str]] = lookup["name_tokens_to_keys"]
        key_to_info: dict[str, dict[str, Any]] = lookup["key_to_info"]

        hits: dict[str, dict[str, Any]] = {}

        def _hit(ik: str, relevance: float, reason: str) -> None:
            if not ik:
                return
            cur = hits.get(ik)
            if cur is None:
                info = dict(key_to_info.get(ik, {}))
                hits[ik] = {"instrument_key": ik, "relevance": float(relevance), "reason": reason, **info}
            else:
                # Keep max relevance and concat reason
                cur["relevance"] = float(max(float(cur.get("relevance") or 0.0), float(relevance)))
                if reason and reason not in str(cur.get("reason") or ""):
                    cur["reason"] = (str(cur.get("reason") or "") + "," + reason).strip(",")

        # Exact symbol matches
        for t in tokens:
            for ik in sym_to_keys.get(t, []):
                _hit(ik, 1.0 if t == (key_to_info.get(ik, {}).get("tradingsymbol") or t) else 0.85, f"sym:{t}")

        # Name-token matches (lower confidence)
        for t in tokens:
            if len(t) >= 4:
                for ik in name_tokens_to_keys.get(t, []):
                    _hit(ik, 0.55, f"name:{t}")

        # Theme hints: treat silver/gold/crude as commodity and prefer MCX instruments
        theme = None
        if "SILVER" in tokens or "XAG" in tokens:
            theme = "SILVER"
        elif "GOLD" in tokens or "XAU" in tokens:
            theme = "GOLD"
        elif "CRUDE" in tokens or "OIL" in tokens:
            theme = "CRUDE"

        out = list(hits.values())
        if theme:
            out.sort(
                key=lambda x: (
                    0 if str(x.get("exchange") or "").upper() == "MCX" else 1,
                    -float(x.get("relevance") or 0.0),
                )
            )
        else:
            out.sort(key=lambda x: -float(x.get("relevance") or 0.0))

        return out[: int(max_mentions)]

    def rebuild_mentions(self, *, limit_news: int = 300) -> dict[str, Any]:
        """Backfill mentions for latest N news items."""
        limit_news = max(1, min(int(limit_news), 2000))
        items = self.latest(limit=limit_news)
        created_ts = int(datetime.now(timezone.utc).timestamp())
        inserted = 0
        scanned = 0
        with db_conn() as conn:
            for it in items:
                scanned += 1
                mentions = self._extract_mentions(it.get("title") or "", it.get("summary"))
                for m in mentions:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO news_mentions (news_url, instrument_key, relevance, reason, created_ts) VALUES (?, ?, ?, ?, ?)",
                            (
                                it.get("url"),
                                m.get("instrument_key"),
                                float(m.get("relevance") or 0.0),
                                (m.get("reason") or None),
                                created_ts,
                            ),
                        )
                        inserted += int(conn.total_changes > 0)
                    except Exception:
                        continue
        return {"ok": True, "scanned": scanned, "inserted": inserted}

    def rescore_recent(self, *, limit_news: int = 300, interval: str = "1d") -> dict[str, Any]:
        """Recompute sentiment/impact for the latest N news items.

        Uses AI-first scoring from extracted mentions. Falls back to keyword heuristic.
        """

        limit_news = max(1, min(int(limit_news), 5000))
        interval = (interval or "1d").strip() or "1d"

        with db_conn() as conn:
            rows = conn.execute(
                "SELECT id, title, summary, sentiment, impact FROM news_items ORDER BY ts DESC LIMIT ?",
                (limit_news,),
            ).fetchall()

        scanned = 0
        updated = 0
        unchanged = 0
        failed = 0

        with db_conn() as conn:
            for r in rows:
                scanned += 1
                try:
                    news_id = int(r[0])
                    title = str(r[1] or "")
                    summary = r[2]

                    mentions: list[dict[str, Any]] = []
                    try:
                        mentions = self._extract_mentions(title, summary)
                    except Exception:
                        mentions = []

                    ai_scored = None
                    try:
                        ai_scored = self._ai_sentiment_from_mentions(mentions, interval=interval)
                    except Exception:
                        ai_scored = None

                    if ai_scored is not None:
                        sent, imp = ai_scored
                    else:
                        sent = _safe_sentiment(title, summary)
                        imp = _impact(sent, title)

                    prev_sent = float(r[3] or 0.0)
                    prev_imp = float(r[4] or 0.0)

                    # Avoid churn from tiny float differences.
                    if abs(prev_sent - float(sent)) < 1e-6 and abs(prev_imp - float(imp)) < 1e-6:
                        unchanged += 1
                        continue

                    conn.execute(
                        "UPDATE news_items SET sentiment=?, impact=? WHERE id=?",
                        (float(sent), float(imp), news_id),
                    )
                    updated += 1
                except Exception:
                    failed += 1
                    continue

        return {
            "ok": True,
            "scanned": scanned,
            "updated": updated,
            "unchanged": unchanged,
            "failed": failed,
            "interval": interval,
        }

    def for_query(self, query: str, *, limit: int = 80, min_ts: int | None = None) -> dict[str, Any]:
        """Return news items relevant to a symbol/theme/instrument_key."""
        q = (query or "").strip()
        if not q:
            return {"query": q, "items": [], "summary": {"sentiment": 0.0, "impact": 0.0, "n": 0}}

        items = self.recent(limit=int(limit), min_ts=min_ts)
        # If query looks like instrument_key, treat it as a key hint.
        key_hint = q if "|" in q else None

        out: list[dict[str, Any]] = []
        for it in items:
            title = it.get("title") or ""
            summary = it.get("summary")

            mentions = self._extract_mentions(title, summary)
            macro_tags = self._tag_macro(title, summary)

            if key_hint:
                keep = any(m.get("instrument_key") == key_hint for m in mentions)
            else:
                up = f"{title} {summary or ''}".upper()
                keep = q.upper() in up
                if not keep:
                    # symbol/theme match through extracted mentions
                    keep = any((q.upper() in (m.get("tradingsymbol") or "")) for m in mentions)

            if keep:
                out.append({**it, "mentions": mentions, "macro": macro_tags})

        # Aggregate impact-weighted sentiment
        if not out:
            return {"query": q, "items": [], "summary": {"sentiment": 0.0, "impact": 0.0, "n": 0}}

        s = sum(float(i.get("sentiment") or 0.0) * float(i.get("impact") or 0.0) for i in out)
        w = sum(float(i.get("impact") or 0.0) for i in out) or 1.0
        summary_obj = {"sentiment": float(s / w), "impact": float(w / len(out)), "n": len(out)}
        return {"query": q, "items": out, "summary": summary_obj}

    def ingest_and_store(
        self,
        *,
        title: str,
        source: str,
        raw_text: str,
        instrument_key: str | None = None,
        url: str | None = None,
    ) -> None:
        title = (title or "").strip()
        source = (source or "").strip()
        raw_text = (raw_text or "").strip()
        if not title or not source or not raw_text:
            raise ValueError("title/source/raw_text are required")

        link = (url or "").strip() or f"manual://{source}/{abs(hash(title))}"
        summary = raw_text
        if instrument_key:
            summary = f"[{instrument_key}] {summary}"

        ts = int(datetime.now(timezone.utc).timestamp())

        mentions: list[dict[str, Any]] = []
        if instrument_key:
            mentions.append({"instrument_key": instrument_key, "relevance": 1.0, "reason": "hint"})
        try:
            mentions.extend(self._extract_mentions(title, summary))
        except Exception:
            pass

        ai_scored = None
        try:
            ai_scored = self._ai_sentiment_from_mentions(mentions, interval="1d")
        except Exception:
            ai_scored = None

        if ai_scored is not None:
            sent, impact = ai_scored
        else:
            sent = _safe_sentiment(title, summary)
            impact = _impact(sent, title)

        with db_conn() as conn:
            cur = conn.execute("SELECT id FROM news_items WHERE url=?", (link,))
            if cur.fetchone() is not None:
                return
            conn.execute(
                "INSERT INTO news_items (ts, source, title, url, summary, sentiment, impact) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, source, title, link, summary, float(sent), float(impact)),
            )

            # Best-effort mentions capture
            try:
                for m in mentions:
                    conn.execute(
                        "INSERT OR IGNORE INTO news_mentions (news_url, instrument_key, relevance, reason, created_ts) VALUES (?, ?, ?, ?, ?)",
                        (link, m.get("instrument_key"), float(m.get("relevance") or 0.0), (m.get("reason") or None), ts),
                    )
            except Exception:
                pass

    def seed_demo_if_empty(self, *, max_items: int = 8) -> int:
        max_items = max(1, min(int(max_items), 50))
        with db_conn() as conn:
            cur = conn.execute("SELECT COUNT(1) AS n FROM news_items")
            n = int(cur.fetchone()["n"])
            if n > 0:
                return 0

        samples = [
            ("Markets steady as investors await policy cues", "DemoWire"),
            ("Large-cap shares lead gains in afternoon trade", "DemoWire"),
            ("Oil prices slip; inflation worries ease", "DemoWire"),
            ("Banking stocks rally on strong earnings", "DemoWire"),
            ("Tech sector mixed amid global cues", "DemoWire"),
            ("Rupee range-bound ahead of key data", "DemoWire"),
            ("Auto stocks rise on demand optimism", "DemoWire"),
            ("Defensive shares outperform in choppy session", "DemoWire"),
        ]

        inserted = 0
        for i, (title, source) in enumerate(samples[:max_items]):
            try:
                self.ingest_and_store(
                    title=title,
                    source=source,
                    raw_text=title,
                    url=f"demo://news/{i}",
                )
                inserted += 1
            except Exception:
                continue
        return inserted

    def fetch_and_store(self, limit_per_feed: int = 20) -> dict:
        urls = _parse_rss_urls()
        if not urls:
            return {"ok": False, "reason": "NEWS_RSS_URLS empty", "inserted": 0}

        inserted = 0
        fetched = 0

        # Try optional feedparser; if missing, just do nothing.
        try:
            import feedparser  # type: ignore
        except Exception:
            return {"ok": False, "reason": "feedparser not installed", "inserted": 0}

        for url in urls:
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue

            source = (getattr(feed, "feed", {}) or {}).get("title") or url
            entries = list(getattr(feed, "entries", []) or [])[: int(limit_per_feed)]
            fetched += len(entries)

            for e in entries:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                summary = (e.get("summary") or e.get("description") or "").strip() or None
                if not title or not link:
                    continue

                # Best-effort time parsing
                ts = int(datetime.now(timezone.utc).timestamp())
                published = e.get("published") or e.get("updated")
                if published:
                    try:
                        # feedparser exposes parsed struct
                        st = e.get("published_parsed") or e.get("updated_parsed")
                        if st:
                            ts = int(datetime(*st[:6], tzinfo=timezone.utc).timestamp())
                    except Exception:
                        pass

                sent = _safe_sentiment(title, summary)
                impact = _impact(sent, title)

                mentions: list[dict[str, Any]] = []
                try:
                    mentions = self._extract_mentions(title, summary)
                except Exception:
                    mentions = []

                try:
                    ai_scored = self._ai_sentiment_from_mentions(mentions, interval="1d")
                    if ai_scored is not None:
                        sent, impact = ai_scored
                except Exception:
                    pass

                with db_conn() as conn:
                    # Deduplicate by url
                    cur = conn.execute("SELECT id FROM news_items WHERE url=?", (link,))
                    if cur.fetchone() is not None:
                        continue
                    conn.execute(
                        "INSERT INTO news_items (ts, source, title, url, summary, sentiment, impact) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ts, source, title, link, summary, float(sent), float(impact)),
                    )
                    try:
                        for m in mentions:
                            conn.execute(
                                "INSERT OR IGNORE INTO news_mentions (news_url, instrument_key, relevance, reason, created_ts) VALUES (?, ?, ?, ?, ?)",
                                (link, m.get("instrument_key"), float(m.get("relevance") or 0.0), (m.get("reason") or None), ts),
                            )
                    except Exception:
                        pass
                    inserted += 1

        return {"ok": True, "fetched": fetched, "inserted": inserted, "feeds": len(urls)}

    def latest(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        with db_conn() as conn:
            cur = conn.execute(
                "SELECT ts, source, title, url, summary, sentiment, impact FROM news_items ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "ts": int(r["ts"]),
                    "source": r["source"],
                    "title": r["title"],
                    "url": r["url"],
                    "summary": r["summary"],
                    "sentiment": float(r["sentiment"]),
                    "impact": float(r["impact"]),
                }
                for r in cur.fetchall()
            ]

    def recent(self, *, limit: int = 50, min_ts: int | None = None) -> list[dict]:
        """Return recent news items.

        Args:
            limit: max number of items.
            min_ts: optional minimum unix timestamp filter.
        """

        limit = max(1, min(int(limit), 200))
        with db_conn() as conn:
            if min_ts is None:
                cur = conn.execute(
                    "SELECT ts, source, title, url, summary, sentiment, impact FROM news_items ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
            else:
                cur = conn.execute(
                    "SELECT ts, source, title, url, summary, sentiment, impact FROM news_items WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                    (int(min_ts), limit),
                )
            return [
                {
                    "ts": int(r["ts"]),
                    "source": r["source"],
                    "title": r["title"],
                    "url": r["url"],
                    "summary": r["summary"],
                    "sentiment": float(r["sentiment"]),
                    "impact": float(r["impact"]),
                }
                for r in cur.fetchall()
            ]

    def market_sentiment(self, lookback: int = 200) -> dict:
        items = self.latest(limit=lookback)
        if not items:
            return {"sentiment": 0.0, "impact": 0.0, "n": 0}
        s = sum(float(i["sentiment"]) * float(i["impact"]) for i in items)
        w = sum(float(i["impact"]) for i in items) or 1.0
        return {"sentiment": float(s / w), "impact": float(w / len(items)), "n": len(items)}

    def match_to_instrument(self, instrument_key: str, limit: int = 50) -> list[dict]:
        """Best-effort mapping: matches token after last '|' against title."""
        token = instrument_key.split("|")[-1].strip().lower()
        if not token:
            return []
        token = re.sub(r"\s+", "", token)
        items = self.latest(limit=limit)
        out: list[dict] = []
        for i in items:
            t = re.sub(r"\s+", "", (i.get("title") or "").lower())
            if token and token in t:
                out.append(i)
        return out
