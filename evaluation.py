import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "overcooked_ai")))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from llm_training import ActorCritic


def evaluate_policy(
    model,
    layout_name="cramped_room",
    num_eval_episodes=20,
    horizon=400,
    device="cpu",
    deterministic=True
):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    
    model.eval()
    episode_returns = []
    soups_delivered = []
    stagnation_steps = []

    for ep in range(num_eval_episodes):
        env.reset()
        ep_reward = 0.0
        zero_reward_count = 0
        done = False
        step = 0

        while not done and step < horizon:
            step += 1
            
            # Encode states for both agents
            lossless_obs = env.lossless_state_encoding_mdp(env.state)
            obs_array = np.array([lossless_obs[i].flatten() for i in range(2)])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)

            with torch.no_grad():
                # Forward pass expected shape match
                hidden = model.fc(obs_tensor)
                logits = model.actor(hidden)
                
                if deterministic:
                    actions_idx = torch.argmax(logits, dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    actions_idx = dist.sample()

            joint_action = tuple(Action.INDEX_TO_ACTION[a.item()] for a in actions_idx)
            
            # Overcooked env step returns (next_state, sparse_reward, done, info)
            next_state, reward, done, info = env.step(joint_action)

            ep_reward += reward
            if reward == 0:
                zero_reward_count += 1

        episode_returns.append(ep_reward)
        soups_delivered.append(ep_reward / 20.0)
        stagnation_steps.append(zero_reward_count)

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_soups = np.mean(soups_delivered)
    avg_stagnation_pct = (np.mean(stagnation_steps) / horizon) * 100

    print("=" * 50)
    print(f"EVALUATION RESULTS ({layout_name}) - {num_eval_episodes} Episodes")
    print("=" * 50)
    print(f"Mean Environment Return : {mean_return:.2f} ± {std_return:.2f}")
    print(f"Mean Soups Delivered    : {mean_soups:.2f}")
    print(f"Stagnation Steps (%)    : {avg_stagnation_pct:.1f}%")
    print("=" * 50)

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_soups": mean_soups,
        "stagnation_pct": avg_stagnation_pct
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with matching dimensions
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=400)
    env.reset()
    sample_obs = env.lossless_state_encoding_mdp(env.state)
    obs_dim = int(np.prod(sample_obs[0].shape))
    action_dim = len(Action.ALL_ACTIONS)

    model = ActorCritic(obs_dim, action_dim).to(device)

    # If you save weights during training using torch.save(model.state_dict(), "model.pt"):
    # model.load_state_dict(torch.load("model.pt", map_location=device))

    # Path to saved weights file
    model_weights_path = "checkpoints/best_model.pt"  # or "checkpoints/model_ep_30.pt"

    if os.path.exists(model_weights_path):
        # Load the state dict into the model architecture
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Successfully loaded weights from {model_weights_path}")
    else:
        print(f"Warning: No checkpoint found at {model_weights_path}. Running with random initialization.")

    evaluate_policy(
        model=model,
        layout_name="cramped_room",
        num_eval_episodes=20,
        horizon=400,
        device=device
    )