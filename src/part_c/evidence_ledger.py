"""Part C — Evidence Ledger & Entity Resolver (§7 of Plan C).

In-memory evidence ledger scoped to a single sample:
- Tracks all entities observed across context, history, and successful tool execution outputs.
- Prevents ID hallucinations and resolves canonical entity IDs.
- Tracks provenance and freshness (marks entities stale/invalidated upon write operations).
- Strict disambiguation: returns None on tie/ambiguity to prevent arbitrary ID guessing.
- Preserves exact opaque IDs byte-for-byte (no lowercase, no stripping).
"""
from __future__ import annotations

import copy
import unicodedata
from typing import Any, Optional


ID_KEY_PATTERNS = {
    "device_id": "device",
    "contact_id": "contact",
    "event_id": "event",
    "reminder_id": "reminder",
    "alarm_id": "alarm",
    "note_id": "note",
    "task_id": "task",
    "bill_id": "bill",
    "ride_id": "ride",
    "track_id": "track",
    "playlist_id": "playlist",
    "account_id": "account",
    "account_number": "account",
    "appliance_id": "appliance",
    "item_id": "item",
    "message_id": "message",
    "chat_id": "chat",
    "room_id": "room",
    "flight_id": "flight",
    "beneficiary_id": "beneficiary",
    "booking_id": "booking",
    "order_id": "order",
}


class EvidenceLedger:
    """Scoped in-memory evidence ledger for a single sample."""

    # Map tool write operations to entity types invalidated
    WRITE_INVALIDATIONS = {
        "smart_home": ["device", "scene"],
        "contacts": ["contact"],
        "calendar": ["event"],
        "reminders": ["reminder"],
        "alarm": ["alarm"],
        "notes": ["note"],
        "tasks": ["task"],
        "appliances": ["appliance"],
        "door_lock": ["device", "lock"],
        "music": ["playlist", "track"],
        "ride_hailing": ["ride"],
        "bill_payment": ["bill"],
        "banking": ["account"],
    }

    def __init__(self, sample: dict):
        self.sample_id = sample.get("id", "")
        self.current_time = sample.get("context", {}).get("current_time", "")
        self.user_profile = sample.get("context", {}).get("user_profile", {})
        self.step_idx = 0

        # {entity_type: {canonical_id: {canonical_id, entity_type, attributes, source_kind, source_ref, observed_step, validity}}}
        self.entities: dict[str, dict[str, dict]] = {
            "device": {},
            "contact": {},
            "event": {},
            "reminder": {},
            "alarm": {},
            "note": {},
            "task": {},
            "bill": {},
            "account": {},
            "ride": {},
            "track": {},
            "playlist": {},
            "appliance": {},
            "other": {},
        }

        # Seed from context and history
        self._seed_from_context(sample)
        self._seed_from_history(sample)

    def _seed_from_context(self, sample: dict):
        """Seed known context information (user profile, locations)."""
        city = self.user_profile.get("city")
        if city and isinstance(city, str):
            self.add_entity(
                entity_type="other",
                canonical_id=city,
                attributes={"type": "user_city", "name": city},
                source_kind="context",
                source_ref="user_profile.city",
                validity="current"
            )

    def _seed_from_history(self, sample: dict):
        """Extract explicit entities mentioned in prior turns in history."""
        history = sample.get("history", [])
        for h_idx, turn in enumerate(history):
            tool_calls = turn.get("tool_calls", [])
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                c_name = tc.get("name", "")
                c_args = tc.get("arguments", {})
                if not isinstance(c_args, dict):
                    continue
                for k, v in c_args.items():
                    if isinstance(v, str) and (k.endswith("_id") or k in ID_KEY_PATTERNS):
                        etype = ID_KEY_PATTERNS.get(k, k.replace("_id", ""))
                        self.add_entity(
                            entity_type=etype,
                            canonical_id=v,
                            attributes={"used_in_history": True, "turn": h_idx},
                            source_kind="history",
                            source_ref=f"turn_{h_idx}.{c_name}",
                            validity="current"
                        )

    def add_entity(
        self,
        entity_type: str,
        canonical_id: str,
        attributes: dict,
        source_kind: str,
        source_ref: str,
        validity: str = "current"
    ):
        """Register or update an entity in the ledger."""
        if not canonical_id or not isinstance(canonical_id, str):
            return
        if entity_type not in self.entities:
            self.entities[entity_type] = {}

        existing = self.entities[entity_type].get(canonical_id, {})
        merged_attrs = copy.deepcopy(existing.get("attributes", {}))
        if isinstance(attributes, dict):
            merged_attrs.update(attributes)

        self.entities[entity_type][canonical_id] = {
            "canonical_id": canonical_id,
            "entity_type": entity_type,
            "attributes": merged_attrs,
            "source_kind": source_kind,
            "source_ref": source_ref,
            "observed_step": self.step_idx,
            "validity": validity,
        }

    def update_from_tool_result(self, tool_name: str, arguments: dict, result: dict):
        """Parse observations from read tools and record ground truth entities."""
        self.step_idx += 1
        if not isinstance(result, dict) or not result.get("ok", True):
            return

        # Recursively ingest any dict or list containing recognizable IDs
        self._ingest_data(result, tool_name)

    def _ingest_data(self, data: Any, tool_name: str, parent_type: Optional[str] = None):
        """Recursively scan tool output for entity objects."""
        if isinstance(data, list):
            for item in data:
                self._ingest_data(item, tool_name, parent_type)
        elif isinstance(data, dict):
            # Check for known ID keys in this dict
            found_id = None
            found_type = None

            for k, etype in ID_KEY_PATTERNS.items():
                if k in data and isinstance(data[k], str) and data[k]:
                    found_id = data[k]
                    found_type = etype
                    break

            if not found_id and "id" in data and isinstance(data["id"], str) and data["id"]:
                found_id = data["id"]
                # Infer type from tool name
                app = tool_name.split(".", 1)[0]
                found_type = self._infer_entity_type_from_app(app) or parent_type or "other"

            if found_id and found_type:
                self.add_entity(
                    entity_type=found_type,
                    canonical_id=found_id,
                    attributes=data,
                    source_kind="tool_result",
                    source_ref=tool_name,
                    validity="current"
                )

            # Recurse into nested dicts/lists
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    inferred_sub = ID_KEY_PATTERNS.get(f"{k}_id", k)
                    self._ingest_data(v, tool_name, inferred_sub)

    @staticmethod
    def _infer_entity_type_from_app(app: str) -> str:
        mapping = {
            "smart_home": "device",
            "contacts": "contact",
            "calendar": "event",
            "reminders": "reminder",
            "alarm": "alarm",
            "notes": "note",
            "tasks": "task",
            "bill_payment": "bill",
            "ride_hailing": "ride",
            "music": "track",
            "banking": "account",
            "appliances": "appliance",
        }
        return mapping.get(app, "other")

    def invalidate_on_write(self, tool_name: str, arguments: dict):
        """Invalidate affected entities after a write operation (§7.2 rule)."""
        app = tool_name.split(".", 1)[0]
        affected_types = self.WRITE_INVALIDATIONS.get(app, [])

        for etype in affected_types:
            if etype in self.entities:
                for cid, entry in self.entities[etype].items():
                    # If specific target ID matches arguments, mark stale
                    arg_id = arguments.get(f"{etype}_id") or arguments.get("id")
                    if arg_id and arg_id == cid:
                        entry["validity"] = "stale"
                    elif not arg_id:
                        # General invalidation for the app
                        entry["validity"] = "unknown"

    def resolve_entity(
        self,
        entity_type: str,
        query_name: Optional[str] = None,
        attributes_filter: Optional[dict] = None
    ) -> Optional[str]:
        """Resolve the canonical ID using multi-attribute matching and validity priority.

        Rules:
        1. Never resolve stale entities.
        2. Exact name/title match gets highest score.
        3. Token overlap gets partial score.
        4. In case of ambiguous tie (two distinct entities with identical top score),
           returns None to avoid arbitrary hallucination.
        """
        table = self.entities.get(entity_type, {})
        if not table:
            return None

        candidates = []

        for cid, entry in table.items():
            if entry["validity"] == "stale":
                continue

            attrs = entry["attributes"]
            score = 0

            if query_name:
                q_norm = unicodedata.normalize("NFC", query_name.strip().lower())
                name_val = str(attrs.get("name") or attrs.get("title") or attrs.get("label") or "")
                n_norm = unicodedata.normalize("NFC", name_val.strip().lower())
                
                if q_norm and n_norm:
                    if q_norm == n_norm:
                        score += 15
                    elif q_norm in n_norm or n_norm in q_norm:
                        score += 8
                    else:
                        # Check word overlap
                        q_words = set(q_norm.split())
                        n_words = set(n_norm.split())
                        overlap = q_words & n_words
                        if overlap:
                            score += len(overlap) * 3

                # Also match against ID itself if relevant
                cid_norm = cid.strip().lower()
                if q_norm == cid_norm:
                    score += 12
                elif q_norm in cid_norm:
                    score += 6

            if attributes_filter:
                for k, v in attributes_filter.items():
                    if attrs.get(k) == v:
                        score += 5

            if entry["validity"] == "current":
                score += 1

            if score > 0:
                candidates.append((score, cid))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])

        # Ambiguity check: if top 2 candidates have the exact same score > 0, return None
        if len(candidates) >= 2 and candidates[0][0] == candidates[1][0] and candidates[0][1] != candidates[1][1]:
            return None

        return candidates[0][1]

    def has_entity(self, entity_type: str, canonical_id: str) -> bool:
        """Check if an entity exists and is current (not stale)."""
        entry = self.entities.get(entity_type, {}).get(canonical_id)
        return entry is not None and entry.get("validity") != "stale"

    def get_known_ids(self) -> set[str]:
        """Return all canonical IDs known in the ledger."""
        ids = set()
        for etype, table in self.entities.items():
            for cid in table:
                ids.add(cid)
        return ids
