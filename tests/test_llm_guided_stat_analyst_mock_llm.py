import importlib.util
import sys
import unittest
from pathlib import Path


def load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "usecase" / "llm_rag_ml" / "testing" / "mock_llm_scenario_runner.py"
    spec = importlib.util.spec_from_file_location("mock_llm_scenario_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load mock_llm_scenario_runner")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestLlmGuidedStatAnalystMockLlm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_device_advertise_scenario(self):
        scenario_file = (
            Path(__file__).resolve().parents[1]
            / "usecase"
            / "llm_rag_ml"
            / "testing"
            / "scenarios"
            / "device_advertise"
            / "scenario.json"
        )
        self.runner.assert_scenario(scenario_file)


if __name__ == "__main__":
    unittest.main()
