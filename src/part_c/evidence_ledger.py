"""Part C — Evidence Ledger & Entity Resolver (§5.1 of Review Plan v3).

In-memory evidence ledger scoped to a single sample:
- Tracks all entities observed across context, history, and tool execution outputs.
- Prevents ID hallucinations and resolves canonical entity IDs (devices, contacts, events, alarms).
- Automatically invalidates entities (marks as stale) upon write operations.
- Provides fallback read/ask recommendations when required IDs are missing.

Usage:
    ledger = EvidenceLedger(sample)
    ledger.update_from_tool_result("contacts.search", {"query": "Mẹ"}, result)
    canonical_id = ledger.resolve_entity(entity_type="contact", query_name="Mẹ")
    ledger.invalidate_on_write("smart_home.light_set", {"device_id": "light_living"})
"""
import copy
from typing import Any, Optional


class EvidenceLedger:
    """Scoped in-memory evidence ledger for one sample."""

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
        "door_lock": ["lock"],
    }

    def __init__(self, sample: dict):
        self.sample_id = sample.get("id", "")
        self.current_time = sample.get("context", {}).get("current_time", "")
        self.user_profile = sample.get("context", {}).get("user_profile", {})
        self.step_idx = 0

        # {entity_type: {canonical_id: {attributes, source_kind, source_ref, observed_step, validity}}}
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
            "other": {},
        }

        # Seed from context and history
        self._seed_from_context(sample)
        self._seed_from_history(sample)

    def _seed_from_context(self, sample: dict):
        """Seed known context information."""
        # City/location
        city = self.user_profile.get("city")
        if city:
            self.add_entity(
                entity_type="other",
                canonical_id=city,
                attributes={"type": "user_city", "name": city},
                source_kind="context",
                source_ref="user_profile.city"
            )

    def _seed_from_history(self, sample: dict):
        """Extract explicit mentions from prior turns in history."""
        history = sample.get("history", [])
        for h_idx, turn in enumerate(history):
            content = turn.get("content", "")
            # Tool results in history
            tool_calls = turn.get("tool_calls", [])
            for tc in tool_calls:
                c_name = tc.get("name", "")
                c_args = tc.get("arguments", {})
                # Extract explicit IDs used in history
                for k, v in c_args.items():
                    if k.endswith("_id") and isinstance(v, str):
                        etype = k.replace("_id", "")
                        self.add_entity(
                            entity_type=etype,
                            canonical_id=v,
                            attributes={"used_in_history": True, "turn": h_idx},
                            source_kind="history",
                            source_ref=f"turn_{h_idx}.{c_name}"
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
        if entity_type not in self.entities:
            self.entities[entity_type] = {}

        existing = self.entities[entity_type].get(canonical_id, {})
        merged_attrs = copy.deepcopy(existing.get("attributes", {}))
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

        # 1. Contacts
        if tool_name in ("contacts.search", "contacts.list", "contacts.get"):
            contacts_list = result.get("contacts") or ([result["contact"]] if "contact" in result else [])
            for c in contacts_list:
                cid = c.get("id") or c.get("contact_id")
                if cid:
                    self.add_entity("contact", cid, c, source_kind="tool_result", source_ref=tool_name)

        # 2. Smart Home
        elif tool_name in ("smart_home.get_status", "smart_home.list_devices"):
            devices = result.get("devices", [])
            for d in devices:
                did = d.get("id") or d.get("device_id")
                if did:
                    self.add_entity("device", did, d, source_kind="tool_result", source_ref=tool_name)

        # 3. Calendar
        elif tool_name in ("calendar.list_events", "calendar.get_event"):
            events = result.get("events") or ([result["event"]] if "event" in result else [])
            for ev in events:
                eid = ev.get("id") or ev.get("event_id")
                if eid:
                    self.add_entity("event", eid, ev, source_kind="tool_result", source_ref=tool_name)

        # 4. Reminders
        elif tool_name in ("reminders.list", "reminders.get"):
            rems = result.get("reminders") or ([result["reminder"]] if "reminder" in result else [])
            for r in rems:
                rid = r.get("id") or r.get("reminder_id")
                if rid:
                    self.add_entity("reminder", rid, r, source_kind="tool_result", source_ref=tool_name)

        # 5. Alarms
        elif tool_name in ("alarm.list", "alarm.get"):
            alarms = result.get("alarms") or ([result["alarm"]] if "alarm" in result else [])
            for a in alarms:
                aid = a.get("id") or a.get("alarm_id")
                if aid:
                    self.add_entity("alarm", aid, a, source_kind="tool_result", source_ref=tool_name)

    def invalidate_on_write(self, tool_name: str, arguments: dict):
        """Invalidate affected entities after a write operation (§5.1 rule)."""
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

        Priority order:
        1. Valid ('current') exact match on name/label.
        2. Valid match with highest attribute overlap.
        3. History reference if no current match.
        """
        table = self.entities.get(entity_type, {})
        if not table:
            return None

        candidates = []
        for cid, entry in table.items():
            if entry["validity"] == "stale":
                continue  # Never resolve stale IDs!

            attrs = entry["attributes"]
            score = 0

            if query_name:
                q_norm = query_name.strip().lower()
                e_name = str(attrs.get("name") or attrs.get("title") or attrs.get("label") or cid).lower()
                if q_norm == e_name:
                    score += 10
                elif q_norm in e_name or e_name in q_norm:
                    score += 5

            if attributes_filter:
                for k, v in attributes_filter.items():
                    if attrs.get(k) == v:
                        score += 3

            if entry["validity"] == "current":
                score += 2

            if score > 0:
                candidates.append((score, cid))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    def has_entity(self, entity_type: str, canonical_id: str) -> bool:
        """Check if an entity exists and is current."""
        entry = self.entities.get(entity_type, {}).get(canonical_id)
        return entry is not None and entry.get("validity") != "stale"

    def get_summary(self) -> dict:
        """Return ledger state summary."""
        return {
            etype: {
                cid: {
                    "validity": data["validity"],
                    "source": data["source_ref"]
                }
                for cid, data in table.items()
            }
            for etype, table in self.entities.items()
            if table
        }
