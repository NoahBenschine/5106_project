from pydantic import BaseModel, Field, ValidationError


class CounterfactualRewardPatch(BaseModel):
    step: int = Field(ge=1, strict=True)
    agent_idx: int = Field(ge=0, strict=True)
    reward_bonus: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
        description="Counterfactual reward adjustment between -1 and 1",
    )
    reasoning: str


class CounterfactualFeedback(BaseModel):
    patches: list[CounterfactualRewardPatch]


def apply_reward_patches(base_rewards, step_indices, patches):
    """Return patched rewards and an audit trail without changing the input.

    Steps are episode-relative, matching the trajectory's step field.
    For duplicate agent-step targets, the first valid patch wins.
    """
    if len(base_rewards) != len(step_indices):
        raise ValueError("Each reward row must have a corresponding step")
    if len(set(step_indices)) != len(step_indices):
        raise ValueError("Step indices must be unique")

    rewards = [list(row) for row in base_rewards]
    num_agents = len(rewards[0]) if rewards else 0
    if any(len(row) != num_agents for row in rewards):
        raise ValueError("Reward rows must have equal lengths")

    step_to_row = {
        step: row for row, step in enumerate(step_indices)
    }
    seen = set()
    audit = []

    for raw_patch in patches:
        try:
            data = (
                raw_patch.model_dump()
                if isinstance(raw_patch, BaseModel)
                else raw_patch
            )
            patch = CounterfactualRewardPatch.model_validate(data)
        except ValidationError:
            audit.append({
                "status": "rejected",
                "reason": "invalid_patch",
            })
            continue

        target = (patch.step, patch.agent_idx)
        rejection = None

        if patch.step not in step_to_row:
            rejection = "step_outside_window"
        elif patch.agent_idx >= num_agents:
            rejection = "invalid_agent"
        elif target in seen:
            rejection = "duplicate_target"

        if rejection:
            audit.append({
                "status": "rejected",
                "step": patch.step,
                "agent_idx": patch.agent_idx,
                "reason": rejection,
            })
            continue

        row = step_to_row[patch.step]
        before = rewards[row][patch.agent_idx]
        after = before + patch.reward_bonus
        rewards[row][patch.agent_idx] = after
        seen.add(target)

        audit.append({
            "status": "applied",
            "step": patch.step,
            "agent_idx": patch.agent_idx,
            "base_reward": before,
            "llm_adjustment": patch.reward_bonus,
            "final_reward": after,
            "reasoning": patch.reasoning,
        })

    return rewards, audit