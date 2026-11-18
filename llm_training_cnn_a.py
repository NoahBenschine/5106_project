import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pydantic import ValidationError
import ollama


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
# 1. Shared CNN Actor-Critic Neural Network (PyTorch)
# ----------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(ActorCritic, self).__init__()
        
        # Spatial feature extraction via Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.action_dim = action_dim
        
    def _build_fc(self, x):
        with torch.no_grad():
            conv_out = self.conv(x)
            flattened_dim = conv_out.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU()
        ).to(x.device)
        self.actor = nn.Linear(128, self.action_dim).to(x.device)
        self.critic = nn.Linear(128, 1).to(x.device)

    def get_action_and_value(self, x, action=None):
            # If input is (Batch, H, W, C) where C is the last dimension:
            # Permute to (Batch, C, H, W) for PyTorch Conv2d
            if x.dim() == 4 and x.shape[-1] == self.conv[0].in_channels:
                x = x.permute(0, 3, 1, 2)
            elif x.dim() == 3:  # Single agent grid (H, W, C) -> (1, C, H, W)
                x = x.permute(2, 0, 1).unsqueeze(0)
    
            # Lazy initialize fully connected heads on first pass to fit grid dimensions
            if not hasattr(self, 'fc'):
                self._build_fc(x)
    
            hidden = self.fc(self.conv(x))
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            
            if action is None:
                action = probs.sample()
                
            return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).squeeze(-1)


# ----------------------------------------------------
# 2. Counterfactual Evaluator & Logging Setup
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
    """Sends recent K steps to Gemini with prompt builder context for counterfactual evaluation."""
    prompt = build_counterfactual_prompt(
        trajectory_window,
        environment_context,
    )
    print(prompt)
    start_time = time.time()
    try:

        options = {
            "num_ctx": 2048,      # Keep context minimal to speed up prompt processing
            "num_predict": 80,    # Cap max output tokens
            "num_thread": 6,      # Match your physical performance cores (avoid hyperthreads)
            "temperature": 0.1,       # Drop from 128 to 64 tokens (66 tokens generated took 4.4s)
            "use_mmap": True,          # Keeps model loaded in RAM
            "use_mlock": True,         # Prevents OS from swapping model memory to disk
            "flash_attn": True,        # Enables fast attention kernels for prefill
        }
        response = ollama.chat(
            model="qwen2.5:1.5b", # or "qwen2.5:7b"
            messages=[{"role": "user", "content": prompt}],
            format="json", # Forces structured JSON output
            keep_alive="24h",  # Keeps model warm in RAM for the entire training run
            options=options
        )

        latency = time.time() - start_time
        # Convert nanoseconds to seconds
        prompt_eval_time = response.get('prompt_eval_duration', 0) / 1e9  # Input processing time
        eval_time = response.get('eval_duration', 0) / 1e9                # Output generation time
        prompt_tokens = response.get('prompt_eval_count', 0)
        eval_tokens = response.get('eval_count', 0)

        print(f"Input Processing: {prompt_eval_time:.2f}s ({prompt_tokens} tokens)")
        print(f"Output Generation: {eval_time:.2f}s ({eval_tokens} tokens @ {eval_tokens/eval_time:.1f} t/s)")
        token_info = {
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": eval_tokens,
            "total_tokens": prompt_tokens + eval_tokens,
        }

        # Extract JSON content from Ollama response
        if hasattr(response, 'message'):
            raw_json = response.message.content
        else:
            raw_json = response['message']['content']
        # Log LLM I/O detail
        write_log(model_io_log_path, {
            "timestamp": time.time(),
            "episode": episode,
            "global_step": global_step,
            "status": "response_received",
            "latency_sec": latency,
            "prompt": prompt,
            "raw_response": raw_json,
            "tokens": token_info
        })

        if not raw_json:
            write_log(model_io_log_path, {
                "timestamp": time.time(),
                "episode": episode,
                "global_step": global_step,
                "status": "empty_response",
            })
            return []

        try:
            feedback = CounterfactualFeedback.model_validate_json(raw_json)
        except ValidationError as error:
            write_log(model_io_log_path, {
                "timestamp": time.time(),
                "episode": episode,
                "global_step": global_step,
                "status": "invalid_feedback",
                "error_message": str(error),
            })
            return []

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
    
    # Map raw action tuples to concise string representations
    ACTION_MAP = {
        (0, -1): "up",
        (0, 1): "down",
        (1, 0): "right",
        (-1, 0): "left",
        (0, 0): "stay",
        "interact": "interact"
    }

    agent_data = {}
    for i, player in enumerate(state.players):
        held = player.held_object.name if player.has_object() else "empty"
        act = joint_action[i] if joint_action else (0, 0)
        act_str = ACTION_MAP.get(act, str(act))
        
        agent_data[f"a{i}"] = {
            "pos": list(player.position),
            "hold": held,
            "act": act_str
        }

    # Only include pots that contain ingredients or are cooking/ready
    active_pots = []
    for pot_pos in mdp.get_pot_locations():
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            active_pots.append({
                "pos": list(pot_pos),
                "items": len(obj.ingredients),
                "ready": getattr(obj, "is_ready", False),
                "cooking": getattr(obj, "is_cooking", False)
            })

    info = info or {}
    shaping = [float(v) for v in info.get("shaped_r_by_agent", [0.0, 0.0])]

    # Streamlined dictionary output
    return {
        "step": step_count,
        "agents": agent_data,
        "pots": active_pots,
        "sparse": float(reward),
        "shaping": shaping
    }

# def extract_rich_step_info(env, joint_action, step_count, sparse_reward, state_before=None, info=None):
#     # 1. Fallback to current env state if state_before wasn't passed
#     state = state_before if state_before is not None else env.state
    
#     # 2. Extract slim agent state from state.players and joint_action
#     slim_agents = {}
#     for idx, player in enumerate(state.players):
#         # Action taken by this specific agent in the joint tuple
#         act = joint_action[idx] if joint_action and len(joint_action) > idx else None
        
#         slim_agents[f"a{idx}"] = {
#             "pos": list(player.position),
#             "dir": list(player.orientation),
#             "act": str(act)
#         }

#     # 3. Extract active pots from state.objects
#     active_pots = {}
#     for pos, obj in state.objects.items():
#         if obj.name == "soup":
#             # Only record pots that have ingredients or are actively cooking
#             num_items = len(obj.ingredients)
#             is_cooking = getattr(obj, "is_cooking", False) or getattr(obj, "is_ready", False)
            
#             if num_items > 0 or is_cooking:
#                 active_pots[f"pot_{pos[0]}_{pos[1]}"] = {
#                     "pos": list(pos),
#                     "items": num_items,
#                     "timer": getattr(obj, "cook_time", 0)
#                 }

#     # 4. Extract environment shaping rewards safely from info
#     shaping = [0.0, 0.0]
#     if info and "shaped_r_by_agent" in info:
#         shaping = [float(r) for r in info["shaped_r_by_agent"]]

#     return {
#         "step": step_count,
#         "agents": slim_agents,
#         "pots": active_pots,
#         "sparse": float(sparse_reward),
#         "shaping": shaping,
#     }

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
    checkpoint_dir="checkpoints/cnn",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gemini_client = genai.Client() if not dry_run else None

    MAX_STEPS_PER_EPISODE = 20 if DEBUG_MODE else horizon
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    environment_context = build_environment_context(mdp, layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    
    action_dim = len(Action.ALL_ACTIONS)
    
    env.reset()
    sample_obs = env.lossless_state_encoding_mdp(env.state)
    obs_shape = sample_obs[0].shape  # (H, W, C)
    in_channels = obs_shape[-1]      # Number of spatial channels
    
    # Initialize CNN Actor-Critic network
    model = ActorCritic(in_channels, action_dim).to(device)

    # Trigger lazy initialization of fc, actor, and critic BEFORE loading weights
    dummy_obs = torch.tensor(np.array([sample_obs[0]]), dtype=torch.float32).to(device)
    with torch.no_grad():
        model.get_action_and_value(dummy_obs)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # ----------------------------------------------------
    # RESUME LOADING LOGIC
    # ----------------------------------------------------
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    start_episode = 0
    best_reward = -float("inf")

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

    for episode in range(start_episode, total_episodes):
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
        obs_buffer = []          # List of spatial tensors shape [num_agents, H, W, C]
        actions_buffer = []      # List of tensors shape [num_agents]
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
            obs_array = np.array([lossless_obs[i] for i in range(num_agents)])
            
            # Keep spatial grid tensor shape (num_agents, H, W, C) for CNN processing
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)

            with torch.no_grad():
                actions, _, _, _ = model.get_action_and_value(obs_tensor)

            joint_action = tuple(Action.INDEX_TO_ACTION[a.item()] for a in actions)
            state_before = snapshot_state(env.state)
            next_state, sparse_reward, done, info = env.step(joint_action)
            episode_env_reward += sparse_reward

            shaped_rewards = info.get("shaped_r_by_agent", [0.0] * num_agents)
            step_base_rewards = [sparse_reward + shaped_rewards[i] for i in range(num_agents)]

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
            time_env_step += (time.time() - t_env_start)
            # lookback_steps = 10
            llm_window_size = 10
            # --- Process Batch & Update Policy Every K Steps (or Episode End) ---
            if step_count % k_steps == 0 or done:
                batch_len = len(obs_buffer)
                recent_window = trajectory_log[-llm_window_size:]

                if verbose_trajectory:
                    print(
                        f"[Trajectory Log - Ep {episode + 1} | "
                        f"Patching Steps {step_indices_buffer[0]}-{step_indices_buffer[-1]} | "
                        f"Context Steps {recent_window[0]['step']}-{recent_window[-1]['step']}]"
                    )
                # --- STEP 2: Query Counterfactual Feedback ---
                t_llm_start = time.time()
                patches = []

                if not dry_run and gemini_client:
                    current_time = time.time()
                    time_since_last_call = current_time - last_gemini_call_time


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
                        next_obs_arr = np.array([next_lossless[i] for i in range(num_agents)])
                        next_obs_tensor = torch.tensor(next_obs_arr, dtype=torch.float32, device=device)
                        _, _, _, next_val = model.get_action_and_value(next_obs_tensor)
                        running_return = next_val.squeeze(-1)

                for t in reversed(range(batch_len)):
                    running_return = batch_rewards[t] + gamma * running_return
                    batch_returns[t] = running_return
                time_return_calc += (time.time() - t_return_start)

                # --- STEP 4: Policy Loss Calculation & Backpropagation ---
                t_update_start = time.time()
                # Batch shape: (batch_len * num_agents, H, W, C)
                flat_obs = batch_obs.view(-1, *obs_shape)
                flat_actions = batch_actions.view(-1)
                flat_returns = batch_returns.view(-1)

                _, log_probs, entropy, values = model.get_action_and_value(flat_obs, action=flat_actions)
                values = values.squeeze(-1)

                advantages = flat_returns - values.detach()
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
        k_steps=100,
        dry_run=False,
        verbose_trajectory=False,
        horizon=400  # Set horizon to standard 400 steps
    )
