import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 0. Import Local Overcooked AI Package
# ----------------------------------------------------
# Ensures the local overcooked_ai directory in the same folder is prioritized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "overcooked_ai")))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

# Gemini SDK Setup
from google import genai
from google.genai import types


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

def query_gemini_counterfactuals(client, trajectory_window):
    """Sends the recent K steps to Gemini to get counterfactual reward adjustments."""
    prompt = f"""
    You are an expert multi-agent RL credit assignment evaluator for Overcooked.
    Analyze this sequence of trajectory steps where two agents interact:
    
    {json.dumps(trajectory_window, indent=2)}
    
    For each agent in the window, evaluate counterfactual actions:
    Did an agent perform a helpful intermediate action (e.g., placing soup ingredient, passing plate)
    that wasn't immediately rewarded by the environment? Or did they block the other agent?
    
    Return reward adjustments (between -1.0 and 1.0) for relevant steps.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CounterfactualFeedback,
                temperature=0.2
            )
        )
        feedback = CounterfactualFeedback.model_validate_json(response.text)
        return feedback.patches
    except Exception as e:
        print(f"[Gemini API Warning] Counterfactual call failed: {e}")
        return []


# ----------------------------------------------------
# 3. Main Training Loop
# ----------------------------------------------------
def train_overcooked_marl(layout_name="cramped_room", total_episodes=200, k_steps=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Gemini Client (requires GEMINI_API_KEY env variable)
    gemini_client = genai.Client()

    # Build Overcooked MDP and Environment
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=400)
    
    num_agents = 2
    action_dim = len(Action.ALL_ACTIONS) # 6 actions: N, S, E, W, Stay, Interact
    
    # Reset environment to infer state shape
    env.reset()
    sample_obs = env.mdp.get_lossless_encoding_dict(env.state)
    obs_shape = sample_obs[0].shape
    obs_dim = int(np.prod(obs_shape)) # Flatten 3D spatial grid tensor
    
    # Single network for both agents (Parameter Sharing)
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(total_episodes):
        env.reset()
        episode_reward = 0
        trajectory_log = []
        step_count = 0
        done = False

        while not done:
            step_count += 1
            
            # Extract state tensors for both agents
            lossless_obs = env.mdp.get_lossless_encoding_dict(env.state)
            obs_array = np.array([lossless_obs[i].flatten() for i in range(num_agents)])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)

            # Get Actions from Model
            with torch.no_grad():
                actions, log_probs, _, values = model.get_action_and_value(obs_tensor)

            # Format actions for Overcooked (Index to Action Object)
            joint_action = tuple(Action.INDEX_TO_ACTION[a.item()] for a in actions)

            # Step Environment
            next_state, reward, done, info = env.step(joint_action)
            episode_reward += reward

            # Per-agent reward tracking
            rewards_list = [float(reward)] * num_agents 

            # Trajectory logging for Gemini
            step_record = {
                "step": step_count,
                "agent_actions": [Action.ACTION_TO_INDEX[a] for a in joint_action],
                "env_reward": float(reward)
            }
            trajectory_log.append(step_record)

            # --------------------------------------------------
            # Gemini Counterfactual Patching (Every K Steps)
            # --------------------------------------------------
            if step_count % k_steps == 0 and len(trajectory_log) >= k_steps:
                recent_window = trajectory_log[-k_steps:]
                patches = query_gemini_counterfactuals(gemini_client, recent_window)
                
                # Apply patches to rewards_list for credit assignment
                for patch in patches:
                    if 0 <= patch.agent_idx < num_agents:
                        rewards_list[patch.agent_idx] += patch.reward_bonus

            # --------------------------------------------------
            # Policy Optimization Step
            # --------------------------------------------------
            reward_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(device)
            advantage = reward_tensor - values.squeeze(-1)

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{total_episodes} | Total Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    train_overcooked_marl(layout_name="cramped_room", total_episodes=100, k_steps=20)