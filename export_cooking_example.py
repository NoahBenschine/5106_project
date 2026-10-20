import json
from pathlib import Path

import llm_training as training
from prompt_builder import (
    build_environment_context,
    build_counterfactual_prompt,
)

folder = Path("evidence/phase4/scripted_cooking")
folder.mkdir(parents=True, exist_ok=True)

mdp = training.OvercookedGridworld.from_layout_name("cramped_room")
env = training.OvercookedEnv.from_mdp(mdp, horizon=100)
env.reset()

directions = {
    "up": (0, -1),
    "down": (0, 1),
    "right": (1, 0),
    "left": (-1, 0),
    "stay": (0, 0),
    "interact": "interact",
}

records = []
stages = []
total_sparse = 0.0


def act(name, stage):
    global total_sparse

    before = training.snapshot_state(env.state)
    joint_action = (directions[name], directions["stay"])
    _, reward, done, info = env.step(joint_action)

    step = len(records) + 1
    record = training.extract_rich_step_info(
        env,
        joint_action,
        step,
        reward,
        state_before=before,
        info=info,
    )
    records.append(record)
    stages.append({"step": step, "stage": stage})
    total_sparse += float(reward)

    assert not done, f"Episode ended during: {stage}"
    return float(reward)


print("Initial state:", env.state)
print("Old dynamics:", mdp.old_dynamics)

act("up", "approach onion dispenser")

for onion in range(1, 4):
    act("left", "face onion dispenser")
    act("interact", f"collect onion {onion}")
    assert env.state.players[0].held_object.name == "onion"

    act("right", "approach pot")
    act("up", "face pot")
    act("interact", f"add onion {onion}")
    assert env.state.players[0].held_object is None

    if onion < 3:
        act("left", "return to onion dispenser")

act("interact", "start cooking")

for name in ("down", "left", "down", "interact"):
    act(name, "collect dish")
assert env.state.players[0].held_object.name == "dish"

for name in ("right", "up", "up"):
    act(name, "return to pot")

while not mdp.soup_ready_at_location(env.state, (2, 0)):
    assert len(records) < 90, "Soup did not become ready in time"
    act("stay", "wait for cooking")

act("interact", "collect ready soup")
assert env.state.players[0].held_object.name == "soup"

for name in ("down", "right", "down"):
    act(name, "approach serving counter")

delivery_reward = act("interact", "deliver soup")
assert delivery_reward == 20.0
assert total_sparse == 20.0
assert env.state.players[0].held_object is None

# Check continuity and reward bookkeeping.
for index, record in enumerate(records):
    if index:
        assert records[index - 1]["state_after"] == record["state_before"]

    rewards = record["rewards"]
    for agent in range(2):
        assert abs(
            rewards["base_training_rewards"][agent]
            - rewards["team_sparse"]
            - rewards["environment_shaping"][agent]
        ) < 1e-8

context = build_environment_context(mdp, "cramped_room")

with (folder / "trajectory.jsonl").open("w") as log:
    for record in records:
        log.write(json.dumps(record, allow_nan=False) + "\n")

(folder / "stages.json").write_text(
    json.dumps(stages, indent=2) + "\n"
)

# Export consecutive 10-step windows, matching our smoke-test window size.
for start in range(0, len(records), 10):
    window = records[start:start + 10]
    label = f"{window[0]['step']:03d}_{window[-1]['step']:03d}"

    (folder / f"prompt_{label}.txt").write_text(
        build_counterfactual_prompt(window, context),
        encoding="utf-8",
    )
    (folder / f"input_{label}.json").write_text(
        json.dumps(
            {"environment": context, "trajectory": window},
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

summary = {
    "status": "PASS",
    "layout": "cramped_room",
    "steps": len(records),
    "total_sparse_reward": total_sparse,
    "delivery_reward": delivery_reward,
    "agent_1_behavior": "stay throughout",
    "initial_state_modified": False,
    "gemini_called": False,
}

(folder / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print(json.dumps(summary, indent=2))
print("Saved to:", folder.resolve())
print("PASS: Cooking trajectory and supervisor prompts exported.")