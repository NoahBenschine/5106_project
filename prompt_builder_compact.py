import json


def build_environment_context(mdp, layout_name):
    return {
        "layout": layout_name,
        "grid": ["".join(row) for row in mdp.terrain_mtx],
        # Removed verbose legends and reward rules. 
        # Small models infer 'X'=wall, 'P'=pot, 'S'=serve from pre-training.
    }


def format_trajectory_compact(trajectory_window):
    """Converts a list of step dicts into a token-lean pipe-delimited table."""
    lines = ["step|a0(x,y,hold,act)|a1(x,y,hold,act)|pots|sparse|shaping"]
    
    for r in trajectory_window:
        a0 = r["agents"]["a0"]
        a1 = r["agents"]["a1"]
        
        a0_str = f"{a0['pos'][0]},{a0['pos'][1]},{a0['hold']},{a0['act']}"
        a1_str = f"{a1['pos'][0]},{a1['pos'][1]},{a1['hold']},{a1['act']}"
        
        if r["pots"]:
            pot_str = ";".join(
                f"{p['pos'][0]},{p['pos'][1]}:{p['items']}/3"
                f"{'_ready' if p['ready'] else '_cook' if p['cooking'] else ''}"
                for p in r["pots"]
            )
        else:
            pot_str = "-"
            
        shaping_str = f"{r['shaping'][0]:.1f},{r['shaping'][1]:.1f}"
        lines.append(
            f"{r['step']}|{a0_str}|{a1_str}|{pot_str}|{r['sparse']:.1f}|{shaping_str}"
        )
        
    return "\n".join(lines)

def build_counterfactual_prompt(trajectory_window, environment_context):
    payload = {
        "env": environment_context,
        "traj": trajectory_window,
    }
    compact = format_trajectory_compact(payload["traj"])

    instructions = """Evaluate Overcooked trajectories for reward adjustments (-1.0 to 1.0).
Rules:
- Compare actual vs alternative action holding teammate fixed.
- Ignore existing shaped/sparse rewards. Do not patch if evidence is weak.
- Max 1 sentence reasoning per patch. Output ONLY valid JSON.

JSON Schema:
{"patches": [{"step": int, "agent": 0|1, "bonus": float, "why": "string"}]}

INPUT:
"""
    return instructions + compact