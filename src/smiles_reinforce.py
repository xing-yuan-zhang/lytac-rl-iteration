from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rdkit import Chem
from rdkit.Chem import QED

from smiles_model import SmilesRNNPolicy
from smiles_data import SmilesVocab
from smiles_rl_env import SmilesRLEnv


def generate_episode(
    policy: SmilesRNNPolicy,
    env: SmilesRLEnv,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> Tuple[List[torch.Tensor], float, Dict]:

    if device is None:
        device = next(policy.parameters()).device
    log_probs: List[torch.Tensor] = []

    obs = env.reset()
    done = False
    final_reward = 0.0
    final_info: Dict = {}
    
    while not done:
        # obs: (max_len,) -> (1, max_len)
        obs_t = torch.tensor(obs, dtype=torch.long, device=device).unsqueeze(0)

        # Forward through policy
        logits, _ = policy(obs_t)  # (1, T, V)

        seq_len = len(env.current_tokens)
        logits_step = logits[:, seq_len - 1, :]  # (1, V)

        if temperature > 0:
            logits_step = logits_step / temperature

        probs = F.softmax(logits_step, dim=-1)  # (1, V)
        dist = Categorical(probs)
        action = dist.sample()                  # (1,)
        log_prob = dist.log_prob(action)        # (1,)

        obs, reward, done, info = env.step(int(action.item()))

        log_probs.append(log_prob.squeeze(0))
        if done:
            final_reward = float(reward)
            final_info = info

    return log_probs, final_reward, final_info


def reinforce_train(
    policy: SmilesRNNPolicy,
    env: SmilesRLEnv,
    num_episodes: int = 5000,
    batch_episodes: int = 16,
    lr: float = 1e-4,
    gamma: float = 1.0,
    reward_scale: float = 1.0,
    baseline_momentum: float = 0.9,
    log_every: int = 50,
    eval_every: int = 200,
    eval_num_samples: int = 256,
    max_len: int = 120,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> SmilesRNNPolicy:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.train()
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    baseline: Optional[float] = None

    episode_rewards = []
    global_step = 0

    for episode_idx in range(1, num_episodes + 1):
        batch_loss = 0.0
        batch_log = []

        for _ in range(batch_episodes):
            log_probs, reward, info = generate_episode(
                policy=policy,
                env=env,
                device=device,
                temperature=1.0,
            )

            scaled_reward = reward * reward_scale

            if baseline is None:
                baseline = scaled_reward
            else:
                baseline = baseline_momentum * baseline + (1 - baseline_momentum) * scaled_reward

            advantage = scaled_reward - baseline

            # REINFORCE loss: - E[(R - b) * log pi(a_t | s_t)]
            log_probs_tensor = torch.stack(log_probs)  # (T,)
            loss_episode = -advantage * log_probs_tensor.mean()

            batch_loss += loss_episode
            episode_rewards.append(reward)

            batch_log.append(
                {
                    "reward": reward,
                    "scaled_reward": scaled_reward,
                    "smiles": info.get("smiles", ""),
                    "is_valid": info.get("is_valid", False),
                    "qed": info.get("qed", 0.0),
                }
            )

        batch_loss = batch_loss / batch_episodes

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

        global_step += 1

        # Logging
        if log_every > 0 and episode_idx % log_every == 0:
            recent_rewards = episode_rewards[-log_every * batch_episodes :]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            print(
                f"[Episode {episode_idx}/{num_episodes} | "
                f"Updates {global_step}] "
                f"Avg reward (last {log_every * batch_episodes} eps): {avg_reward:.4f}, "
                f"Baseline: {0.0 if baseline is None else baseline:.4f}, "
                f"Batch loss: {batch_loss.item():.4f}"
            )

        # Evaluation
        if eval_every > 0 and episode_idx % eval_every == 0:
            metrics = evaluate_policy(
                policy=policy,
                vocab=env.vocab,
                num_samples=eval_num_samples,
                max_len=max_len,
                device=device,
            )
            print(
                f"  [Eval @ Episode {episode_idx}] "
                f"Avg QED: {metrics['avg_qed']:.4f}, "
                f"Validity: {metrics['validity'] * 100:.2f}%, "
                f"Novelty: {metrics['novelty'] * 100:.2f}%"
            )
            if save_path is not None:
                torch.save(policy.state_dict(), save_path)
                print(f"  Saved policy checkpoint to {save_path}")
            
        policy.train()

    return policy


@torch.no_grad()
def evaluate_policy(
    policy: SmilesRNNPolicy,
    vocab: SmilesVocab,
    num_samples: int = 256,
    max_len: int = 120,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:

    if device is None:
        device = next(policy.parameters()).device

    policy.eval()

    smiles_list: List[str] = []
    qeds: List[float] = []
    valid_flags: List[bool] = []

    for _ in range(num_samples):
        s = policy.sample(
            vocab=vocab,
            max_len=max_len,
            greedy=False,
            device=device,
        )
        smiles_list.append(s)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            valid_flags.append(False)
            qeds.append(0.0)
        else:
            valid_flags.append(True)
            qeds.append(float(QED.qed(mol)))

    avg_qed = float(np.mean(qeds)) if qeds else 0.0
    validity = float(np.mean(valid_flags)) if valid_flags else 0.0

    unique_smiles = set(smiles_list)
    novelty = len(unique_smiles) / max(1, len(smiles_list))

    return {
        "avg_qed": avg_qed,
        "validity": validity,
        "novelty": novelty,
    }
