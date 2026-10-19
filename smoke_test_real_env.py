import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import llm_training as training


root = Path(tempfile.mkdtemp(
    prefix="real_smoke_",
    dir="evidence/phase4",
))
logs = root / "logs"

# Fail immediately if this test attempts to create a Gemini client.
with patch.object(
    training.genai,
    "Client",
    side_effect=AssertionError("Gemini must remain disabled"),
):
    training.train_overcooked_marl(
        layout_name="cramped_room",
        total_episodes=1,
        k_steps=10,
        horizon=20,
        dry_run=True,
        verbose_trajectory=False,
        max_episode_seconds=300,
        log_dir=str(logs),
        checkpoint_dir=str(root / "checkpoints"),
    )


def read_jsonl(name):
    return [
        json.loads(line)
        for line in (logs / name).read_text().splitlines()
    ]


trajectory = read_jsonl("trajectory_log.jsonl")
audits = read_jsonl("reward_patch_audit.jsonl")
performance = read_jsonl("performance_log.jsonl")

assert len(trajectory) == 20, "Expected 20 real environment steps"
assert len(audits) == 2, "Expected two reward-processing windows"
assert len(performance) == 2, "Expected two policy updates"

assert [entry["step_indices"] for entry in audits] == [
    list(range(1, 11)),
    list(range(11, 21)),
]

for entry in audits:
    assert entry["llm_enabled"] is False
    assert entry["base_rewards"] == entry["final_rewards"]
    assert entry["applied_count"] == 0
    assert entry["rejected_count"] == 0

for entry in performance:
    for metric in ("actor_loss", "critic_loss", "total_loss"):
        assert math.isfinite(entry[metric]), f"Nonfinite {metric}"

print("PASS: 20 real environment steps and two policy updates")
print("PASS: Base rewards preserved with Gemini disabled")
print("PASS: Logged losses are finite")
print("Evidence folder:", root.resolve())