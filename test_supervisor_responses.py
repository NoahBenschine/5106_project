import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import llm_training as training
from reward_patching import apply_reward_patches


class SupervisorResponseTests(unittest.TestCase):
    def check_response(self, text=None, error=None):
        generate = Mock()

        if error is not None:
            generate.side_effect = error
        else:
            generate.return_value = SimpleNamespace(
                text=text,
                usage_metadata=None,
            )

        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate)
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model_io.jsonl"

            with patch.object(
                training,
                "build_counterfactual_prompt",
                return_value="Offline test prompt",
            ):
                patches = training.query_gemini_counterfactuals(
                    client=client,
                    trajectory_window=[{"step": 1}],
                    model_io_log_path=str(path),
                    episode=1,
                    global_step=1,
                    environment_context={},
                )

            entries = [
                json.loads(line)
                for line in path.read_text().splitlines()
            ]

        generate.assert_called_once()
        self.assertEqual(patches, [])

        base = [[20.0, 20.0]]
        final, audit = apply_reward_patches(base, [1], patches)
        self.assertEqual(final, base)
        self.assertEqual(audit, [])

        return entries

    def test_valid_empty_patches(self):
        entries = self.check_response('{"patches": []}')
        self.assertEqual(
            [entry["status"] for entry in entries],
            ["response_received", "validated"],
        )
        self.assertEqual(entries[-1]["patch_count"], 0)

    def test_empty_response(self):
        for text in (None, "", "   "):
            with self.subTest(text=text):
                entries = self.check_response(text)
                self.assertEqual(
                    [entry["status"] for entry in entries],
                    ["response_received", "empty_response"],
                )

    def test_invalid_feedback(self):
        for text in (
            "this is not JSON",
            '{"patches": [{}]}',
        ):
            with self.subTest(text=text):
                entries = self.check_response(text)
                self.assertEqual(
                    [entry["status"] for entry in entries],
                    ["response_received", "invalid_feedback"],
                )

    def test_api_error(self):
        error = training.APIError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "Synthetic test rate limit",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )
        entries = self.check_response(error=error)
        self.assertEqual(
            [entry["status"] for entry in entries],
            ["rate_limited_or_api_error"],
        )


if __name__ == "__main__":
    unittest.main()