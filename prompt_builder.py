import json


def build_environment_context(mdp, layout_name):
    return {
        "layout_name": layout_name,
        "terrain_rows": ["".join(row) for row in mdp.terrain_mtx],
        "coordinates": (
            "[x, y], zero-based; x increases right, y increases down"
        ),
        "terrain_legend": {
            "X": "counter or wall; not walkable",
            " ": "walkable floor",
            "O": "onion dispenser",
            "T": "tomato dispenser",
            "D": "dish dispenser",
            "P": "cooking pot",
            "S": "serving counter",
        },
        "action_legend": {
            "(0, -1)": "up",
            "(0, 1)": "down",
            "(1, 0)": "right",
            "(-1, 0)": "left",
            "(0, 0)": "stay",
            "interact": "interact with the tile being faced",
        },
        "agent_identity": (
            "agent_0 is players[0]; agent_1 is players[1]. "
            "Keep these identities fixed across steps."
        ),
        "reward_rule": (
            "Each agent's base training reward equals team_sparse "
            "plus that agent's environment_shaping reward. "
            "An LLM bonus is added afterward."
        ),
    }


def build_counterfactual_prompt(trajectory_window, environment_context):
    payload = {
        "environment": environment_context,
        "trajectory": trajectory_window,
    }
    compact = json.dumps(payload, separators=(",", ":"), allow_nan=False)

    instructions = """You evaluate cooperative Overcooked trajectories.

Use state_before, the joint actions, and state_after to assess each transition.
The agents and pots summaries describe the state AFTER the joint action.
Read actions from agents.agent_0.action and agents.agent_1.action.
Use state_before to determine each agent's starting position and orientation.
A directional action can change orientation even when movement fails.
Evaluate interact using the agent's orientation in state_before.
A movement conflict alone does not identify which agent deserves a penalty.
Rewards include shared sparse reward and individual environment shaping.

For a proposed adjustment:
- Identify the actual agent, episode-relative step, and observed action effect.
- State a specific feasible alternative action for that agent.
- Compare likely team progress under the actual and alternative actions,
  holding the teammate's action fixed for that transition.
- Ground the comparison in the supplied layout and state transitions.
- Treat this as an estimate: no alternative trajectory was simulated.
- Explain why an adjustment adds useful credit beyond rewards already given.
- Do not automatically reward an action already covered by shaping.
- Zero sparse reward alone is not evidence of inactivity or obstruction.
- Waiting for cooking or moving toward a task can be useful.
- Do not infer blocking solely from an unchanged position.
- Do not invent events or outcomes beyond the supplied window.
- If the evidence is insufficient, omit the patch.

Return JSON with a patches list. Each patch must contain:
step: an episode-relative step appearing in this window.
agent_idx: 0 or 1.
reward_bonus: a finite number from -1.0 through 1.0.
reasoning: observed evidence, alternative action, expected difference,
and justification considering existing rewards.

Return at most one patch per agent-step pair.
An empty patches list is valid.

INPUT:
"""
    return instructions + compact