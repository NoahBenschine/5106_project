import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from reward_patching import (
    CounterfactualRewardPatch,
    CounterfactualFeedback,
    apply_reward_patches,
)
from prompt_builder import (
    build_environment_context,
    build_counterfactual_prompt,
)

# ----------------------------------------------------
# 0. Import Local Overcooked AI Package
# ----------------------------------------------------
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "overcooked_ai", "src")
    ),
)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

# Gemini SDK Setup
from google import genai
from google.genai import types
from google.genai.errors import APIError


# ----------------------------------------------------
# 1. Shared Actor-Critic Neural Network (PyTorch)
# ----------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def get_action_and_value(self, x, action=None):
        hidden = self.fc(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


# ----------------------------------------------------
# 2. Gemini Counterfactual Evaluator Setup
# ----------------------------------------------------


def write_log(filepath, data):
    """Appends a single JSON entry as a new line in a file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

def query_gemini_counterfactuals(
    client,
    trajectory_window,
    model_io_log_path,
    episode,
    global_step,
    environment_context=None,
):
    """Sends recent K steps to Gemini for counterfactual reward adjustments and logs I/O."""
    prompt = build_counterfactual_prompt(
        trajectory_window,
        environment_context,
    )

    start_time = time.time()
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",  # Updated to Flash Lite for higher daily limits (500 RPD)
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CounterfactualFeedback,
                temperature=0.2,
            ),
        )

        latency = time.time() - start_time
        usage = response.usage_metadata
        token_info = {
            "prompt_tokens": usage.prompt_token_count if usage else 0,
            "candidates_tokens": usage.candidates_token_count if usage else 0,
            "total_tokens": usage.total_token_count if usage else 0,
        }

        # Log LLM I/O detail
        write_log(model_io_log_path, {
            "timestamp": time.time(),
            "episode": episode,
            "global_step": global_step,
            "status": "response_received",
            "latency_sec": latency,
            "prompt": prompt,
            "raw_response": response.text,
            "tokens": token_info
        })

        if not response.text:
            print("[Gemini API Warning] Empty response returned.")
            return []

        feedback = CounterfactualFeedback.model_validate_json(response.text)
        write_log(model_io_log_path, {
            "timestamp": time.time(),
            "episode": episode,
            "global_step": global_step,
            "status": "validated",
            "patch_count": len(feedback.patches),
        })
        return feedback.patches

    except APIError as e:
        print(f"[Gemini API Rate Limit / Error] {e}")
        write_log(model_io_log_path, {
            "timestamp": time.time(),
            "episode": episode,
            "global_step": global_step,
            "status": "rate_limited_or_api_error",
            "error_message": str(e),
            "prompt": prompt
        })
        return []
    except Exception as e:
        print(f"[Gemini API Exception] {e}")
        write_log(model_io_log_path, {
            "timestamp": time.time(),
            "episode": episode,
            "global_step": global_step,
            "status": "exception",
            "error_message": str(e),
            "prompt": prompt
        })
        return []

def snapshot_state(state):
    """Create an independent, JSON-compatible state snapshot."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    return json.loads(json.dumps(state.to_dict(), default=convert))

def extract_rich_step_info(
    env,
    joint_action,
    step_count,
    reward,
    state_before=None,
    info=None,
):
    state = env.state
    mdp = env.mdp
    
    action_names = [str(a) for a in joint_action]
    
    agent_data = {}
    for i, player in enumerate(state.players):
        held_obj = player.held_object.name if player.has_object() else "empty"
        agent_data[f"agent_{i}"] = {
            "pos": list(player.position),
            "orientation": list(player.orientation),
            "holding": held_obj,
            "action": action_names[i]
        }
    
    pot_states = []
    pot_locations = mdp.get_pot_locations()
    
    for pot_pos in pot_locations:
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            pot_states.append({
                "pos": list(pot_pos),
                "is_empty": False,
                "soup_type": obj.name,
                "num_ingredients": len(obj.ingredients),
                "is_cooking": obj.is_cooking if hasattr(obj, "is_cooking") else False,
                "is_ready": obj.is_ready if hasattr(obj, "is_ready") else False
            })
        else:
            pot_states.append({
                "pos": list(pot_pos),
                "is_empty": True,
                "num_ingredients": 0
            })

        info = info or {}
    shaping = [
        float(value)
        for value in info.get("shaped_r_by_agent", [0.0, 0.0])
    ]
    individual_sparse = info.get("sparse_r_by_agent")

    return {
        "schema_version": 2,
        "step": step_count,
        "env_reward": float(reward),
        "agents": agent_data,
        "pots": pot_states,
        "state_before": state_before,
        "state_after": snapshot_state(env.state),
        "rewards": {
            "team_sparse": float(reward),
            "individual_sparse": (
                [float(value) for value in individual_sparse]
                if individual_sparse is not None else None
            ),
            "environment_shaping": shaping,
            "base_training_rewards": [
                float(reward) + value for value in shaping
            ],
        },
        "agent_infos": info.get("agent_infos", []),
    }


# Environment / Execution Config
DEBUG_MODE = False

MIN_GEMINI_INTERVAL = 6.0


# ----------------------------------------------------
# 3. Main Training Loop
# ----------------------------------------------------
def train_overcooked_marl(
    layout_name="cramped_room",
    total_episodes=200,
    k_steps=400,
    dry_run=True,
    verbose_trajectory=True,
    max_episode_seconds=45.0,
    log_dir="logs",
    checkpoint_dir="checkpoints",
    horizon=400,
    gamma=0.99,            # Discount factor for return computation
    entropy_coef=0.01      # Encourages exploration
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    trajectory_log_path = os.path.join(log_dir, "trajectory_log.jsonl")
    model_io_log_path = os.path.join(log_dir, "model_io_log.jsonl")
    performance_log_path = os.path.join(log_dir, "performance_log.jsonl")
    patch_audit_log_path = os.path.join(log_dir, "reward_patch_audit.jsonl")
    
    trajectory_log_path = os.path.join(log_dir, "trajectory_log.jsonl")
    model_io_log_path = os.path.join(log_dir, "model_io_log.jsonl")
    patch_audit_log_path = os.path.join(log_dir, "reward_patch_audit.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gemini_client = genai.Client() if not dry_run else None

    MAX_STEPS_PER_EPISODE = 20 if DEBUG_MODE else horizon
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    environment_context = build_environment_context(mdp, layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    
    num_agents = 2
    action_dim = len(Action.ALL_ACTIONS)
    
    env.reset()
    sample_obs = env.lossless_state_encoding_mdp(env.state)
    obs_shape = sample_obs[0].shape
    obs_dim = int(np.prod(obs_shape))
    
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Resume weights from checkpoint if available
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"--> Resumed training from checkpoint: {best_model_path}")

    max_steps_per_episode = MAX_STEPS_PER_EPISODE
    last_gemini_call_time = 0
    global_step = 0
    best_reward = -float("inf")

    for episode in range(total_episodes):
        env.reset()
        episode_env_reward = 0.0
        episode_total_reward = 0.0
        step_count = 0
        episode_start_time = time.time()
        done = False

        # Rollout buffers
        obs_buffer = []          # List of tensors shape [2, obs_dim]
        actions_buffer = []      # List of tensors shape [2]
        rewards_buffer = []      # List of lists [agent_0_rew, agent_1_rew]
        step_indices_buffer = [] # Step numbers
        trajectory_log = []

        print(f"Episode {episode + 1}")

        while not done:
            step_count += 1
            global_step += 1

          

            if time.time() - episode_start_time > max_episode_seconds:
                print(f"[Timeout] Episode {episode + 1} exceeded {max_episode_seconds}s limit.")
                break

            lossless_obs = env.lossless_state_encoding_mdp(env.state)
            obs_array = np.array([lossless_obs[i].flatten() for i in range(num_agents)])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)

            with torch.no_grad():
                actions, _, _, _ = model.get_action_and_value(obs_tensor)

            joint_action = tuple(Action.INDEX_TO_ACTION[a.item()] for a in actions)
            state_before = snapshot_state(env.state)
            next_state, sparse_reward, done, info = env.step(joint_action)
            episode_env_reward += sparse_reward

            shaped_rewards = info.get("shaped_r_by_agent", [0.0, 0.0])
            step_base_rewards = [sparse_reward + shaped_rewards[0], sparse_reward + shaped_rewards[1]]

            obs_buffer.append(obs_tensor)
            actions_buffer.append(actions)
            rewards_buffer.append(step_base_rewards)
            step_indices_buffer.append(step_count)

            step_record = extract_rich_step_info(
                env,
                joint_action,
                step_count,
                sparse_reward,
                state_before=state_before,
                info=info,
            )
            trajectory_log.append(step_record)

            write_log(trajectory_log_path, {
                "episode": episode + 1,
                "global_step": global_step,
                "record": step_record
            })

            # --- Process Batch & Update Policy Every K Steps (or Episode End) ---
            if step_count % k_steps == 0 or done or step_count >= max_steps_per_episode:
                recent_window = trajectory_log[-len(obs_buffer):]

                if verbose_trajectory:
                    print(f"\n[Trajectory Log - Ep {episode + 1} | Steps {step_indices_buffer[0]}-{step_indices_buffer[-1]}]")
                    print(json.dumps(recent_window, indent=2))

                # 1. Query Gemini, validate patches, and audit rewards.
                patches = []

                if not dry_run and gemini_client:
                    current_time = time.time()
                    time_since_last_call = (
                        current_time - last_gemini_call_time
                    )

                    if time_since_last_call < MIN_GEMINI_INTERVAL:
                        time.sleep(
                            MIN_GEMINI_INTERVAL - time_since_last_call
                        )

                    patches = query_gemini_counterfactuals(
                        gemini_client,
                        recent_window,
                        model_io_log_path,
                        episode + 1,
                        global_step,
                        environment_context=environment_context,
                    )
                    last_gemini_call_time = time.time()

                base_rewards = [list(row) for row in rewards_buffer]

                rewards_buffer, patch_audit = apply_reward_patches(
                    base_rewards,
                    step_indices_buffer,
                    patches,
                )

                write_log(patch_audit_log_path, {
                    "timestamp": time.time(),
                    "episode": episode + 1,
                    "global_step": global_step,
                    "llm_enabled": not dry_run,
                    "step_indices": list(step_indices_buffer),
                    "base_rewards": base_rewards,
                    "final_rewards": rewards_buffer,
                    "patch_audit": patch_audit,
                    "applied_count": sum(
                        item["status"] == "applied"
                        for item in patch_audit
                    ),
                    "rejected_count": sum(
                        item["status"] == "rejected"
                        for item in patch_audit
                    ),
                })

                # 2. Vectorize buffers into single batched tensors
                # Shapes: batch_obs -> [K, 2, obs_dim], batch_actions -> [K, 2], batch_rewards -> [K, 2]
                batch_obs = torch.stack(obs_buffer).to(device)
                batch_actions = torch.stack(actions_buffer).to(device)
                batch_rewards = torch.tensor(rewards_buffer, dtype=torch.float32, device=device)

                batch_len = len(obs_buffer)

                # 3. Compute discounted return targets (R_t) backward across the buffer
                batch_returns = torch.zeros_like(batch_rewards)
                running_return = torch.zeros(num_agents, device=device)

                # Bootstrapping: If not done, estimate future value using critic on the next observation
                if not done:
                    with torch.no_grad():
                        next_lossless = env.lossless_state_encoding_mdp(env.state)
                        next_obs_arr = np.array([next_lossless[i].flatten() for i in range(num_agents)])
                        next_obs_tensor = torch.tensor(next_obs_arr, dtype=torch.float32, device=device)
                        _, _, _, next_val = model.get_action_and_value(next_obs_tensor)
                        running_return = next_val.squeeze(-1)  # shape [2]

                for t in reversed(range(batch_len)):
                    running_return = batch_rewards[t] + gamma * running_return
                    batch_returns[t] = running_return

                # 4. Single Vectorized Forward Pass over Entire K-step Batch
                # Reshape to 2D tensors for batch processing: [K * 2, feature_dim]
                flat_obs = batch_obs.view(-1, obs_dim)
                flat_actions = batch_actions.view(-1)
                flat_returns = batch_returns.view(-1)

                _, log_probs, entropy, values = model.get_action_and_value(flat_obs, action=flat_actions)
                values = values.squeeze(-1)

                # Advantage calculation against discounted return targets
                advantages = flat_returns - values.detach()

                # Calculate vectorized losses
                actor_loss = -(log_probs * advantages).mean()
                critic_loss = nn.functional.mse_loss(values, flat_returns)
                entropy_loss = -entropy_coef * entropy.mean()

                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # 5. Backpropagation step
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                episode_total_reward += float(batch_rewards.sum().item())

                # Log step performance
                write_log(performance_log_path, {
                    "timestamp": time.time(),
                    "episode": episode + 1,
                    "global_step": global_step,
                    "env_reward": float(sparse_reward),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "total_loss": float(total_loss.item())
                })

                # Clear rollout buffers
                obs_buffer.clear()
                actions_buffer.clear()
                rewards_buffer.clear()
                step_indices_buffer.clear()
            if max_steps_per_episode and step_count >= max_steps_per_episode:
                            print(f"[Step Limit] Episode {episode + 1} reached max step limit ({max_steps_per_episode}).")
                            break

        print(f"Episode {episode + 1}/{total_episodes} | Env Reward: {episode_env_reward:.2f} | Total Reward (Shaped + LLM): {episode_total_reward:.2f}")

        # Model Checkpointing
        if (episode + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_ep_{episode + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)

        if episode_env_reward > best_reward:
            best_reward = episode_env_reward
            torch.save(model.state_dict(), best_model_path)
       


if __name__ == "__main__":
    train_overcooked_marl(
        layout_name="cramped_room",
        total_episodes=5,
        k_steps=50,
        dry_run=False,
        verbose_trajectory=True,
        horizon=50
    )