import json
from pathlib import Path

import llm_training as training
from prompt_builder import (
    build_environment_context,
    build_counterfactual_prompt,
)

root = Path("evidence/phase4")
source = root / "real_smoke_ak2hd7bd/logs/trajectory_log.jsonl"

entries = [
    json.loads(line)
    for line in source.read_text().splitlines()
]
window = [entry["record"] for entry in entries[:10]]

assert len(window) == 10
assert all(record["schema_version"] == 2 for record in window)

mdp = training.OvercookedGridworld.from_layout_name("cramped_room")
context = build_environment_context(mdp, "cramped_room")
prompt = build_counterfactual_prompt(window, context)

destination = root / "prompt_example"
destination.mkdir(exist_ok=True)

(destination / "prompt.txt").write_text(prompt, encoding="utf-8")
(destination / "input.json").write_text(
    json.dumps(
        {"environment": context, "trajectory": window},
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)
(destination / "source.txt").write_text(
    str(source) + "\n",
    encoding="utf-8",
)

print("Source:", source)
print("Window steps:", [record["step"] for record in window])
print("Prompt characters:", len(prompt))
print("Saved to:", destination.resolve())
print("No Gemini request was made.")