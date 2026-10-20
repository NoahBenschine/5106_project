import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 0. Import Local Overcooked AI Package
# ----------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "overcooked_ai")))

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
class CounterfactualRewardPatch(BaseModel):
    step: int
    agent_idx: int
    reward_bonus: float = Field(description="Counterfactual reward modifier between -1.0 and 1.0")
    reasoning: str = Field(description="Explanation of the counterfactual evaluation")

class CounterfactualFeedback(BaseModel):
    patches: list[CounterfactualRewardPatch]


def write_log(filepath, data):
    """Appends a single JSON entry as a new line in a file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")



def format_trajectory_to_natural_language(trajectory_window):
    """Converts raw step JSON objects into a concise, token-efficient natural language string."""
    lines = []
    for step_data in trajectory_window:
        step_num = step_data["step"]
        r_env = step_data.get("env_reward", 0.0)
        
        # Format Agents
        ag_strings = []
        for ag_key, ag in step_data["agents"].items():
            ag_idx = ag_key.split("_")[-1]
            holding = ag.get("holding", "empty")
            action = ag.get("action", "stay")
            pos = tuple(ag.get("pos", [0, 0]))
            ag_strings.append(f"Ag{ag_idx} at {pos} holding {holding} took action '{action}'")
        
        agents_str = "; ".join(ag_strings)

        # Format Pots (Only include active/cooking/ready pots to save tokens)
        pot_strings = []
        for pot in step_data.get("pots", []):
            if not pot.get("is_empty", True):
                p_pos = tuple(pot["pos"])
                n_ing = pot.get("num_ingredients", 0)
                status = "ready" if pot.get("is_ready") else ("cooking" if pot.get("is_cooking") else f"{n_ing}/3 ingr")
                pot_strings.append(f"Pot at {p_pos}: {status}")
        
        pots_str = f" | Pots: {', '.join(pot_strings)}" if pot_strings else ""
        lines.append(f"Step {step_num} (Env Reward: {r_env}): {agents_str}{pots_str}")

    return "\n".join(lines)

    y

def query_gemini_counterfactuals(client, trajectory_window, model_io_log_path, episode, global_step):
    """Sends recent K steps formatted as compact natural language lines to Gemini."""
    
    # Convert window to natural language string
    compact_trajectory_str = format_trajectory_to_natural_language(trajectory_window)
    prompt = f"""Analyze this sequence of Overcooked agent trajectory steps:

{compact_trajectory_str}

Task: Evaluate counterfactual actions for each agent.
- Assign a positive bonus (+0.1 to +1.0) if an agent performed a helpful intermediate action (e.g., placing an ingredient in a pot, fetching a plate) that was not rewarded by the environment.
- Assign a negative penalty (-0.1 to -1.0) if an agent blocked the other agent or acted passively.
- Ignore routine movement steps. Return reward adjustments only for steps with notable contribution or hindrance."""

    start_time = time.time()
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CounterfactualFeedback,
                temperature=0.1,
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
            "status": "success",
            "latency_sec": latency,
            "prompt": prompt,
            "raw_response": response.text,
            "tokens": token_info
        })

        if not response.text:
            print("[Gemini API Warning] Empty response returned.")
            return []

        feedback = CounterfactualFeedback.model_validate_json(response.text)
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


def extract_rich_step_info(env, joint_action, step_count, reward):
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

    return {
        "step": step_count,
        "env_reward": float(reward),
        "agents": agent_data,
        "pots": pot_states
    }


# Environment / Execution Config
DEBUG_MODE = False
MIN_GEMINI_INTERVAL = 6.0

SAVE_INTERVAL = 25
# ----------------------------------------------------
# 3. Main Training Loop
# ----------------------------------------------------
def train_overcooked_marl(
    layout_name="cramped_room",
    total_episodes=200,
    k_steps=400,
    num_agents=2,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gemini_client = genai.Client() if not dry_run else None

    MAX_STEPS_PER_EPISODE = 20 if DEBUG_MODE else horizon
    
    # Configure reward shaping parameters explicitly for sub-task feedback
    reward_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3.0,
        "DISH_PICKUP_REWR": 3.0,
        "SOUP_PICKUP_REWR": 5.0
    }
    mdp = OvercookedGridworld.from_layout_name(layout_name, reward_shaping_params=reward_shaping_params)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    
    num_agents = 3
    action_dim = len(Action.ALL_ACTIONS)
    
    env.reset()
    sample_obs = env.lossless_state_encoding_mdp(env.state)
    obs_shape = sample_obs[0].shape
    obs_dim = int(np.prod(obs_shape))
    
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # ----------------------------------------------------
    # PLACE 1: RESUME LOADING LOGIC (Before Episode Loop)
    # ----------------------------------------------------
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    start_episode = 0
    best_reward = -float("inf")

    # Check and load from the latest continuous training state if available
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_reward = checkpoint.get('best_reward', -float("inf"))
        start_episode = checkpoint.get('episode', 0)
        print(f"--> Resumed training from Episode {start_episode} with Best Reward: {best_reward:.2f}")

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

        # Profiling accumulators per episode (in seconds)
        time_env_step = 0.0
        time_llm_eval = 0.0
        time_return_calc = 0.0
        time_policy_update = 0.0

        # Rollout buffers
        obs_buffer = []          # List of tensors shape [2, obs_dim]
        actions_buffer = []      # List of tensors shape [2]
        rewards_buffer = []      # List of lists [agent_0_rew, agent_1_rew]
        step_indices_buffer = [] # Step numbers
        trajectory_log = []

        print(f"\n--- Starting Episode {episode + 1}/{total_episodes} ---")

        while not done:
            step_count += 1
            global_step += 1

            if time.time() - episode_start_time > max_episode_seconds:
                print(f"[Timeout] Episode {episode + 1} exceeded {max_episode_seconds}s limit.")
                break

            # --- STEP 1: Environment Rollout Execution ---
            t_env_start = time.time()
            lossless_obs = env.lossless_state_encoding_mdp(env.state)
            print(lossless_obs)
            print("type:", type(lossless_obs))
            print("len/shape:", len(lossless_obs) if isinstance(lossless_obs, (list, tuple)) else lossless_obs.shape)
            obs_array = np.array([lossless_obs[i].flatten() for i in range(num_agents)])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)

            with torch.no_grad():
                actions, _, _, _ = model.get_action_and_value(obs_tensor)

            joint_action = tuple(Action.INDEX_TO_ACTION[a.item()] for a in actions)
            next_state, sparse_reward, done, info = env.step(joint_action)
            episode_env_reward += sparse_reward

            shaped_rewards = info.get("shaped_r_by_agent", [0.0, 0.0])
            step_base_rewards = [sparse_reward + shaped_rewards[0], sparse_reward + shaped_rewards[1]]

            obs_buffer.append(obs_tensor)
            actions_buffer.append(actions)
            rewards_buffer.append(step_base_rewards)
            step_indices_buffer.append(step_count)

            step_record = extract_rich_step_info(env, joint_action, step_count, sparse_reward)
            trajectory_log.append(step_record)

            write_log(trajectory_log_path, {
                "episode": episode + 1,
                "global_step": global_step,
                "record": step_record
            })
            time_env_step += (time.time() - t_env_start)

            # --- Process Batch & Update Policy Every K Steps (or Episode End) ---
            if step_count % k_steps == 0 or done or step_count >= max_steps_per_episode:
                recent_window = trajectory_log[-len(obs_buffer):]

                if verbose_trajectory:
                    print(f"[Trajectory Log - Ep {episode + 1} | Steps {step_indices_buffer[0]}-{step_indices_buffer[-1]}]")

                # --- STEP 2: Query Gemini for Counterfactual Feedback ---
                t_llm_start = time.time()
                if not dry_run and gemini_client:
                    current_time = time.time()
                    time_since_last_call = current_time - last_gemini_call_time
                    print(current_time)
                    if time_since_last_call < MIN_GEMINI_INTERVAL:
                        time.sleep(MIN_GEMINI_INTERVAL - time_since_last_call)
                    new_time = time.time()
                    print(new_time)
                    print("time spent sleeping:")
                    print(new_time - current_time)
                    patches = query_gemini_counterfactuals(
                        gemini_client, recent_window, model_io_log_path, episode + 1, global_step
                    )
                    last_gemini_call_time = time.time()
                    print("Time taken for gemini counterfactual:")
                    print(last_gemini_call_time - new_time)
                    # Apply LLM patches to historical buffer steps
                    window_start_step = step_indices_buffer[0]
                    for patch in patches:
                        buffer_idx = patch.step - window_start_step
                        if 0 <= buffer_idx < len(rewards_buffer):
                            if 0 <= patch.agent_idx < num_agents:
                                rewards_buffer[buffer_idx][patch.agent_idx] += patch.reward_bonus
                time_llm_eval += (time.time() - t_llm_start)

                # --- STEP 3: Return & Advantage Computation ---
                t_return_start = time.time()
                batch_obs = torch.stack(obs_buffer).to(device)
                batch_actions = torch.stack(actions_buffer).to(device)
                batch_rewards = torch.tensor(rewards_buffer, dtype=torch.float32, device=device)

                batch_len = len(obs_buffer)
                batch_returns = torch.zeros_like(batch_rewards)
                running_return = torch.zeros(num_agents, device=device)

                if not done:
                    with torch.no_grad():
                        next_lossless = env.lossless_state_encoding_mdp(env.state)
                        next_obs_arr = np.array([next_lossless[i].flatten() for i in range(num_agents)])
                        next_obs_tensor = torch.tensor(next_obs_arr, dtype=torch.float32, device=device)
                        _, _, _, next_val = model.get_action_and_value(next_obs_tensor)
                        running_return = next_val.squeeze(-1)

                for t in reversed(range(batch_len)):
                    running_return = batch_rewards[t] + gamma * running_return
                    batch_returns[t] = running_return
                time_return_calc += (time.time() - t_return_start)

                # --- STEP 4: Policy Loss Calculation & Backpropagation ---
                t_update_start = time.time()
                flat_obs = batch_obs.view(-1, obs_dim)
                flat_actions = batch_actions.view(-1)
                flat_returns = batch_returns.view(-1)

                _, log_probs, entropy, values = model.get_action_and_value(flat_obs, action=flat_actions)
                values = values.squeeze(-1)

                advantages = flat_returns - values.detach()

                actor_loss = -(log_probs * advantages).mean()
                critic_loss = nn.functional.mse_loss(values, flat_returns)
                entropy_loss = -entropy_coef * entropy.mean()

                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                episode_total_reward += float(batch_rewards.sum().item())
                time_policy_update += (time.time() - t_update_start)

                # Log step performance
                write_log(performance_log_path, {
                    "timestamp": time.time(),
                    "episode": episode + 1,
                    "global_step": global_step,
                    "env_reward": float(sparse_reward),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "total_loss": float(total_loss.item()),
                    "time_metrics": {
                        "env_step_sec": time_env_step,
                        "llm_eval_sec": time_llm_eval,
                        "return_calc_sec": time_return_calc,
                        "policy_update_sec": time_policy_update
                    }
                })

                # Clear rollout buffers
                obs_buffer.clear()
                actions_buffer.clear()
                rewards_buffer.clear()
                step_indices_buffer.clear()

            if max_steps_per_episode and step_count >= max_steps_per_episode:
                print(f"[Step Limit] Episode {episode + 1} reached max step limit ({max_steps_per_episode}).")
                break

        # Display Summary Timing Info for Episode
        total_ep_time = time.time() - episode_start_time
        print(f"\n--- Episode {episode + 1} Summary ---")
        print(f"Env Reward: {episode_env_reward:.2f} | Total Reward (Shaped + LLM): {episode_total_reward:.2f}")
        print(f"Total Time: {total_ep_time:.2f}s")
        print(f"  ├─ Step 1 (Env Rollout Execution)     : {time_env_step:.4f}s ({time_env_step/total_ep_time*100:.1f}%)")
        print(f"  ├─ Step 2 (Gemini LLM Credit Feedback): {time_llm_eval:.4f}s ({time_llm_eval/total_ep_time*100:.1f}%)")
        print(f"  ├─ Step 3 (Returns & Advantage Calc) : {time_return_calc:.4f}s ({time_return_calc/total_ep_time*100:.1f}%)")
        print(f"  └─ Step 4 (Backprop & Policy Update)  : {time_policy_update:.4f}s ({time_policy_update/total_ep_time*100:.1f}%)")

        # 1. Periodically save latest training state for seamless resuming
        if (episode + 1) % SAVE_INTERVAL == 0 or (episode + 1) == total_episodes:
            torch.save({
                'episode': episode + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_reward': best_reward
            }, latest_checkpoint_path)
            print(f" Saved training state to {latest_checkpoint_path}")

        # 2. Save best performing model ONLY when environment reward achieves a new peak
        if episode_env_reward > best_reward:
            print(f" All-time high reward! ({best_reward:.2f} -> {episode_env_reward:.2f}). Updating best_model.pt...")
            best_reward = episode_env_reward
            torch.save({
                'episode': episode + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_reward': best_reward
            }, best_model_path)


if __name__ == "__main__":
    train_overcooked_marl(
        layout_name="cramped_room",
        total_episodes=5,
        k_steps=400,
        dry_run=False,
        verbose_trajectory=False,
        horizon=400  # Set horizon to standard 400 steps
    )