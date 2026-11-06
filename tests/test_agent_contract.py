"""Tests for Agent Contract & Protocols (§10 of Plan C)."""
import json
import unittest
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.part_c.prompt import parse_agent_response, build_user_prompt
from src.part_c.tool_filter import filter_tools_for_sample
from src.part_c.agent import BaseAgent, RuleBasedAgent


class MockEnv:
    """Mock ToolEnv recording calls and validations."""
    def __init__(self):
        self.validations = []
        self.calls = []
        self._state_hash = "a" * 64

    def validate(self, name, arguments=None):
        self.validations.append((name, arguments or {}))
        if name == "uninstalled.tool":
            return {"ok": False, "error": "tool not installed"}
        return {"ok": True, "error": None}

    def call(self, name, arguments=None):
        self.calls.append({"name": name, "arguments": arguments or {}})
        if name == "sys.ask_missing_info":
            return {"ok": True, "waiting_for_user": True}
        return {"ok": True, "result": "mock_result"}

    def get_tool_calls(self):
        return self.calls

    def get_state_hash(self):
        return self._state_hash


class MockLLMAgent(BaseAgent):
    """Mock agent returning predefined responses."""
    def __init__(self, responses: list[str]):
        super().__init__(max_steps=5, max_repairs=1)
        self.responses = list(responses)
        self.step_counter = 0

    def generate(self, messages: list[dict], tools=None) -> str:
        if self.step_counter < len(self.responses):
            resp = self.responses[self.step_counter]
            self.step_counter += 1
            return resp
        return "DONE"


class TestAgentContract(unittest.TestCase):

    def test_parse_native_tool_call(self):
        resp = '<tool_call>\n{"name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}}\n</tool_call>'
        parsed = parse_agent_response(resp)
        self.assertEqual(parsed["action"], "call")
        self.assertEqual(parsed["name"], "smart_home.light_set")
        self.assertEqual(parsed["arguments"]["device_id"], "light_kitchen")
        self.assertEqual(parsed["arguments"]["power"], "off")

    def test_parse_legacy_call(self):
        resp = 'CALL: {"name": "contacts.search", "arguments": {"query": "Mẹ"}}'
        parsed = parse_agent_response(resp)
        self.assertEqual(parsed["action"], "call")
        self.assertEqual(parsed["name"], "contacts.search")
        self.assertEqual(parsed["arguments"]["query"], "Mẹ")

    def test_parse_done(self):
        self.assertEqual(parse_agent_response("DONE")["action"], "done")
        self.assertEqual(parse_agent_response("done")["action"], "done")
        self.assertEqual(parse_agent_response("<done>")["action"], "done")
        self.assertEqual(parse_agent_response("")["action"], "done")

    def test_tool_filter_installed_apps(self):
        sample = {
            "context": {
                "installed_apps": {
                    "alarm": {"tools": ["alarm.create", "alarm.list"]}
                }
            }
        }
        all_specs = {
            "alarm.create": {"name": "alarm.create"},
            "alarm.list": {"name": "alarm.list"},
            "music.play": {"name": "music.play"},
            "sys.ask_missing_info": {"name": "sys.ask_missing_info"},
        }
        filtered = filter_tools_for_sample(sample, all_specs)
        self.assertIn("alarm.create", filtered)
        self.assertIn("alarm.list", filtered)
        self.assertIn("sys.ask_missing_info", filtered)
        self.assertNotIn("music.play", filtered)

    def test_ask_missing_info_is_terminal(self):
        """sys.ask_missing_info must terminate execution immediately after calling."""
        env = MockEnv()
        sample = {
            "id": "sample_01",
            "context": {
                "installed_apps": {
                    "sms": {"tools": ["sms.send"]}
                }
            }
        }
        responses = [
            '<tool_call>\n{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["recipient"]}}\n</tool_call>',
            '<tool_call>\n{"name": "sms.send", "arguments": {"recipient": "John", "content": "hi"}}\n</tool_call>',
        ]
        agent = MockLLMAgent(responses)
        agent.run("nhắn tin cho tôi", sample, env)

        self.assertEqual(len(env.calls), 1)
        self.assertEqual(env.calls[0]["name"], "sys.ask_missing_info")

    def test_loop_guard(self):
        """Calling identical tool with identical args >= 2 times must trigger loop guard."""
        env = MockEnv()
        sample = {
            "id": "sample_02",
            "context": {
                "installed_apps": {
                    "smart_home": {"tools": ["smart_home.get_status"]}
                }
            }
        }
        call_str = '<tool_call>\n{"name": "smart_home.get_status", "arguments": {}}\n</tool_call>'
        responses = [call_str, call_str, call_str]
        agent = MockLLMAgent(responses)
        agent.run("bật đèn", sample, env)

        # Loop guard stops after the 2nd identical call
        self.assertEqual(len(env.calls), 2)


if __name__ == "__main__":
    unittest.main()
