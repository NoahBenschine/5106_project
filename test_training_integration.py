import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import llm_training as training


class FakeMDP:
    @classmethod
    def from_layout_name(cls, name):
        return cls()


class FakeEnv:
    @classmethod
    def from_mdp(cls, mdp, horizon):
        return cls()

    def reset(self):
        self.step_number = 0
        self.state = 0

    def lossless_state_encoding_mdp(self, state):
        return [
            np.array([state, 0, 1], dtype=np.float32),
            np.array([state, 1, 0], dtype=np.float32),
        ]

    def step(self, actions):
        self.step_number += 1
        self.state = self.step_number
        reward = 20.0 if self.step_number == 4 else 0.0
        return self.state, reward, self.step_number == 4, {
            "shaped_r_by_agent": [0.0, 0.0]
        }


class TrainingIntegrationTests(unittest.TestCase):
    def test_mock_feedback_reaches_learner(self):
        reward_batches = []
        critic_targets = []

        original_tensor = torch.tensor
        original_mse = training.nn.functional.mse_loss

        def capture_tensor(data, *args, **kwargs):
            result = original_tensor(data, *args, **kwargs)
            if isinstance(data, list) and result.shape == (2, 2):
                reward_batches.append(result.detach().cpu().tolist())
            return result

        def capture_mse(prediction, target, *args, **kwargs):
            critic_targets.append(target.detach().cpu().tolist())
            return original_mse(prediction, target, *args, **kwargs)

        def fake_generate_content(**kwargs):
            prompt = kwargs["contents"]
            second_window = '"step":3' in prompt
            step = 3 if second_window else 1
            agent = 1 if second_window else 0
            bonus = -0.25 if second_window else 0.75

            def item(s, a, b):
                return {
                    "step": s,
                    "agent_idx": a,
                    "reward_bonus": b,
                    "reasoning": "Synthetic integration-test feedback",
                }

            return SimpleNamespace(
                text=json.dumps({
                    "patches": [
                        item(step, agent, bonus),
                        item(step, agent, 0.5),  # Duplicate.
                        item(step, 2, 0.5),      # Invalid agent.
                        item(999, 0, 0.5),       # Outside this window.
                    ]
                }),
                usage_metadata=None,
            )

        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=fake_generate_content
            )
        )

        def fake_record(env, actions, step_count, reward):
            return {"step": step_count, "env_reward": float(reward)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"

            with (
                patch.object(training, "OvercookedGridworld", FakeMDP),
                patch.object(training, "OvercookedEnv", FakeEnv),
                patch.object(training, "extract_rich_step_info", fake_record),
                patch.object(training.genai, "Client", return_value=client),
                patch.object(training, "MIN_GEMINI_INTERVAL", 0.0),
                patch.object(training.torch.cuda, "is_available", return_value=False),
                patch.object(training.torch, "tensor", side_effect=capture_tensor),
                patch.object(
                    training.nn.functional,
                    "mse_loss",
                    side_effect=capture_mse,
                ),
            ):
                training.train_overcooked_marl(
                    total_episodes=1,
                    k_steps=2,
                    horizon=4,
                    dry_run=False,
                    verbose_trajectory=False,
                    max_episode_seconds=300,
                    log_dir=str(logs),
                    checkpoint_dir=str(root / "checkpoints"),
                )

            self.assertEqual(reward_batches, [
                [[0.75, 0.0], [0.0, 0.0]],
                [[0.0, -0.25], [20.0, 20.0]],
            ])

            # Terminal window: discounted patched rewards reach the critic.
            np.testing.assert_allclose(
                critic_targets[-1],
                [19.8, 19.55, 20.0, 20.0],
                rtol=1e-5,
            )

            audit = [
                json.loads(line)
                for line in (logs / "reward_patch_audit.jsonl")
                .read_text().splitlines()
            ]
            self.assertEqual(len(audit), 2)
            self.assertEqual(
                [entry["step_indices"] for entry in audit],
                [[1, 2], [3, 4]],
            )

            for entry in audit:
                self.assertEqual(entry["applied_count"], 1)
                self.assertEqual(entry["rejected_count"], 3)
                self.assertEqual(
                    {item.get("reason") for item in entry["patch_audit"]
                     if item["status"] == "rejected"},
                    {"duplicate_target", "invalid_agent", "step_outside_window"},
                )

            model_io = [
                json.loads(line)
                for line in (logs / "model_io_log.jsonl")
                .read_text().splitlines()
            ]
            self.assertEqual(len(model_io), 2)
            self.assertTrue(
                all(entry["status"] == "success" for entry in model_io)
            )

            destination = Path("evidence/phase4/mock_integration")
            destination.mkdir(parents=True, exist_ok=True)
            for name in (
                "reward_patch_audit.jsonl",
                "model_io_log.jsonl",
                "performance_log.jsonl",
                "trajectory_log.jsonl",
            ):
                shutil.copy2(logs / name, destination / name)

        print("Verified: mock JSON -> validated rewards -> critic targets")


if __name__ == "__main__":
    unittest.main()