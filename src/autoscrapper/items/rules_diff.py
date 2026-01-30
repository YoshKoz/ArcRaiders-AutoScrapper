from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RuleChange:
    item_id: str
    name: str
    before_action: str
    after_action: str
    reasons: List[str]


def _normalize_action(value: object) -> str:
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw:
            return raw
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                return entry.strip().lower()
    return ""


def _item_key(item: dict) -> Optional[str]:
    item_id = item.get("id")
    if isinstance(item_id, str) and item_id.strip():
        return f"id:{item_id.strip().lower()}"
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        return f"name:{name.strip().lower()}"
    return None


def _build_index(items: List[dict]) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _item_key(item)
        if key:
            index[key] = item
    return index


def collect_rule_changes(
    default_payload: dict, updated_payload: dict
) -> List[RuleChange]:
    default_items = default_payload.get("items")
    updated_items = updated_payload.get("items")
    if not isinstance(default_items, list) or not isinstance(updated_items, list):
        return []

    default_index = _build_index(default_items)
    changes: List[RuleChange] = []

    for item in updated_items:
        if not isinstance(item, dict):
            continue
        key = _item_key(item)
        if not key:
            continue
        default_item = default_index.get(key)
        if default_item is None and key.startswith("id:"):
            name_key = _item_key({"name": item.get("name")})
            if name_key:
                default_item = default_index.get(name_key)
        if default_item is None:
            continue

        before_action = _normalize_action(default_item.get("action"))
        if not before_action:
            before_action = _normalize_action(default_item.get("decision"))
        after_action = _normalize_action(item.get("action"))
        if not after_action:
            after_action = _normalize_action(item.get("decision"))

        if not before_action or not after_action or before_action == after_action:
            continue

        reasons_raw = item.get("analysis")
        reasons: List[str] = []
        if isinstance(reasons_raw, list):
            for reason in reasons_raw:
                if isinstance(reason, str) and reason.strip():
                    reasons.append(reason.strip())

        item_id = item.get("id") or default_item.get("id") or ""
        name = item.get("name") or default_item.get("name") or item_id
        changes.append(
            RuleChange(
                item_id=str(item_id),
                name=str(name),
                before_action=before_action,
                after_action=after_action,
                reasons=reasons,
            )
        )

    changes.sort(key=lambda change: change.name.lower())
    return changes
