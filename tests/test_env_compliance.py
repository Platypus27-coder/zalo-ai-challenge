"""Tests for Environment & SDK Compliance (§2 and §10 of Plan C)."""
import unittest
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT


class TestEnvCompliance(unittest.TestCase):

    def setUp(self):
        # Ensure train split zaic_tools is in sys.path
        train_dir = DATA_ROOT / "train"
        if str(train_dir) not in sys.path:
            sys.path.insert(0, str(train_dir))

    def test_single_toolenv_instance(self):
        """ToolEnv must be instantiated cleanly per sample."""
        from zaic_tools import ToolEnv
        
        # Load sample train_0001
        env = ToolEnv("train", "train_0001", root=str(DATA_ROOT))
        self.assertIsNotNone(env)
        
        # Must support the 4 participant APIs
        self.assertTrue(hasattr(env, "validate"))
        self.assertTrue(hasattr(env, "call"))
        self.assertTrue(hasattr(env, "get_tool_calls"))
        self.assertTrue(hasattr(env, "get_state_hash"))

    def test_oos_sample_has_valid_initial_state_hash(self):
        """Even with 0 tool calls, env.get_state_hash() must return valid 64-char hex SHA256."""
        from zaic_tools import ToolEnv
        
        env = ToolEnv("train", "train_0001", root=str(DATA_ROOT))
        tool_calls = env.get_tool_calls()
        state_hash = env.get_state_hash()

        self.assertEqual(len(tool_calls), 0)
        self.assertIsInstance(state_hash, str)
        self.assertEqual(len(state_hash), 64)
        # Verify hex format
        int(state_hash, 16)

    def test_uninstalled_tool_rejected_by_validate(self):
        """Validating a tool not in installed_apps must return ok=False."""
        from zaic_tools import ToolEnv
        
        env = ToolEnv("train", "train_0001", root=str(DATA_ROOT))
        
        # Check a completely fake tool
        val = env.validate("non_existent_app.fake_tool", {})
        self.assertFalse(val["ok"])
        self.assertIsNotNone(val.get("error"))

    def test_sys_ask_missing_info_always_available(self):
        """sys.ask_missing_info is always allowed in any sample environment."""
        from zaic_tools import ToolEnv
        
        env = ToolEnv("train", "train_0001", root=str(DATA_ROOT))
        val = env.validate("sys.ask_missing_info", {"tool_name": "alarm.create", "params": ["time"]})
        self.assertTrue(val["ok"])


if __name__ == "__main__":
    unittest.main()
