"""Tests for Evidence Ledger & Grounding (§10 of Plan C)."""
import unittest
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.part_c.evidence_ledger import EvidenceLedger
from src.part_c.task_tracker import TaskTracker


class TestGrounding(unittest.TestCase):

    def test_ledger_entity_ingestion(self):
        sample = {
            "id": "sample_grounding_01",
            "context": {"user_profile": {"city": "Hồ Chí Minh"}},
            "history": []
        }
        ledger = EvidenceLedger(sample)

        # Ingest smart home devices from tool result
        mock_result = {
            "ok": True,
            "devices": [
                {"device_id": "light_kitchen_01", "name": "Đèn nhà bếp", "room": "Bếp"},
                {"device_id": "fan_living_01", "name": "Quạt phòng khách", "room": "Phòng khách"}
            ]
        }
        ledger.update_from_tool_result("smart_home.get_status", {}, mock_result)

        # Resolve kitchen light
        resolved = ledger.resolve_entity(entity_type="device", query_name="Đèn nhà bếp")
        self.assertEqual(resolved, "light_kitchen_01")

        # Resolve fan living room
        resolved_fan = ledger.resolve_entity(entity_type="device", query_name="Quạt phòng khách")
        self.assertEqual(resolved_fan, "fan_living_01")

    def test_ledger_ambiguity_returns_none(self):
        """When two entities have the exact same match score, resolver must return None instead of guessing."""
        sample = {"id": "sample_ambiguous"}
        ledger = EvidenceLedger(sample)

        mock_contacts = {
            "ok": True,
            "contacts": [
                {"contact_id": "c_001", "name": "Nguyễn Văn A"},
                {"contact_id": "c_002", "name": "Nguyễn Văn A"},
            ]
        }
        ledger.update_from_tool_result("contacts.search", {"query": "Nguyễn Văn A"}, mock_contacts)

        # Should return None due to ambiguous tie
        resolved = ledger.resolve_entity(entity_type="contact", query_name="Nguyễn Văn A")
        self.assertIsNone(resolved)

    def test_ledger_freshness_invalidation_on_write(self):
        """Write operation must invalidate affected entities and mark them stale."""
        sample = {"id": "sample_invalidation"}
        ledger = EvidenceLedger(sample)

        mock_alarms = {
            "ok": True,
            "alarms": [
                {"alarm_id": "alarm_0700", "time": "07:00", "label": "Báo thức sáng"}
            ]
        }
        ledger.update_from_tool_result("alarm.list", {}, mock_alarms)
        self.assertTrue(ledger.has_entity("alarm", "alarm_0700"))

        # Delete alarm
        ledger.invalidate_on_write("alarm.delete", {"alarm_id": "alarm_0700"})

        # Entity should now be stale
        self.assertFalse(ledger.has_entity("alarm", "alarm_0700"))
        # Resolving it should now return None
        self.assertIsNone(ledger.resolve_entity("alarm", query_name="Báo thức sáng"))

    def test_opaque_id_preservation(self):
        """Opaque IDs with mixed case, underscores or special chars must be preserved exactly."""
        sample = {"id": "sample_opaque"}
        ledger = EvidenceLedger(sample)

        complex_id = "Device_ABC_123_Kitchen"
        mock_res = {
            "ok": True,
            "devices": [
                {"device_id": complex_id, "name": "Đèn bếp"}
            ]
        }
        ledger.update_from_tool_result("smart_home.get_status", {}, mock_res)
        resolved = ledger.resolve_entity("device", "Đèn bếp")
        self.assertEqual(resolved, complex_id)
        # Verify byte-for-byte exactness
        self.assertEqual(resolved, "Device_ABC_123_Kitchen")


if __name__ == "__main__":
    unittest.main()
