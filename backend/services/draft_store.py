"""In-memory draft store for staged generate → adapt/optimize flows."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

# Drafts expire after 6 hours of inactivity
_TTL_SEC = 6 * 3600
_DRAFTS: Dict[str, Dict[str, Any]] = {}


def _gc() -> None:
    now = time.time()
    dead = [
        k
        for k, v in _DRAFTS.items()
        if now - float(v.get("updated_at") or 0) > _TTL_SEC
    ]
    for k in dead:
        _DRAFTS.pop(k, None)


def save_draft(payload: Dict[str, Any]) -> str:
    _gc()
    draft_id = str(uuid.uuid4())
    now = time.time()
    _DRAFTS[draft_id] = {
        **payload,
        "draft_id": draft_id,
        "created_at": now,
        "updated_at": now,
    }
    return draft_id


def get_draft(draft_id: str) -> Optional[Dict[str, Any]]:
    _gc()
    d = _DRAFTS.get(draft_id)
    if not d:
        return None
    if time.time() - float(d.get("updated_at") or 0) > _TTL_SEC:
        _DRAFTS.pop(draft_id, None)
        return None
    return d


def update_draft(draft_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    d = get_draft(draft_id)
    if not d:
        return None
    d.update(kwargs)
    d["updated_at"] = time.time()
    return d


def delete_draft(draft_id: str) -> None:
    _DRAFTS.pop(draft_id, None)
