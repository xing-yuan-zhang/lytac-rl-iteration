from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rdkit import Chem
from rdkit.Chem import QED

from smiles_model import SmilesRNNPolicy
from smiles_data import SmilesVocab
from smiles_rl_env import SmilesRLEnv


class SmilesActorCritic(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(input_ids)         # (B, T, E)
        output, _ = self.lstm(emb)             # (B, T, H)
        logits = self.policy_head(output)      # (B, T, V)
        values = self.value_head(output).squeeze(-1)  # (B, T)
        return logits, values

    @classmethod
    def from_pretrained_policy(
        cls,
        policy: SmilesRNNPolicy,
        value_init_scale: float = 0.01,
        dropout: float = 0.1,
    ) -> "SmilesActorCritic":
        vocab_size = policy.vocab_size
        embed_dim = policy.embedding.weight.shape[1]
        hidden_dim = policy.lstm.hidden_size
        num_layers = policy.lstm.num_layers

        ac = cls(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        ac.embedding.load_state_dict(policy.embedding.state_dict())
        ac.lstm.load_state_dict(policy.lstm.state_dict())
        ac.policy_head.load_state_dict(policy.output_head.state_dict())

        nn.init.uniform_(ac.value_head.weight, -value_init_scale, value_init_scale)
        nn.init.zeros_(ac.value_head.bias)

        return ac

    @torch.no_grad()
    def sample(
        self,
        vocab: SmilesVocab,
        max_len: int = 120,
        greedy: bool = False,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> str:
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        start_id = vocab.start_id
        end_id = vocab.end_id
        pad_id = vocab.pad_id

        input_ids = torch.tensor([[start_id]], dtype=torch.long, device=device)
        generated_ids: List[int] = [start_id]

        for _ in range(max_len - 1):
            logits, _ = self.forward(input_ids)  # (1, T, V)
            seq_len = input_ids.size(1)
            logits_step = logits[:, seq_len - 1, :]  # (1, V)

            if temperature > 0:
                logits_step = logits_step / temperature

            probs = torch.softmax(logits_step, dim=-1)
            if greedy:
                next_token = torch.argmax(probs, dim=-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            token_id = int(next_token.item())
            generated_ids.append(token_id)

            if token_id == end_id:
                break

            input_ids = torch.cat(
                [input_ids, next_token.view(1, 1)], dim=1
            )  # (1, T+1)

        smiles = vocab.decode(generated_ids, skip_special=True)
        return smiles


def _collect_rollout(
    actor_critic: SmilesActorCritic,
    env: SmilesRLEnv,
    episodes_per_update: int,
    max_len: int,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    obs_buf = []
    actions_buf = []
    logprobs_buf = []
    values_buf = []
    returns_buf = []

    actor_critic.eval()

    for _ in range(episodes_per_update):
        obs = env.reset()
        done = False

        episode_obs = []
        episode_actions = []
        episode_logprobs = []
        episode_values = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, values = actor_critic(obs_t)  # logits: (1, T, V), values: (1, T)

                seq_len = len(env.current_tokens)
                logits_step = logits[:, seq_len - 1, :]    # (1, V)
                value_step = values[:, seq_len - 1]        # (1,)

                if temperature > 0:
                    logits_step = logits_step / temperature

                probs = torch.softmax(logits_step, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()                     # (1,)
                log_prob = dist.log_prob(action)           # (1,)

            next_obs, reward, done, info = env.step(int(action.item()))

            episode_obs.append(obs_t.squeeze(0))       # (T,)
            episode_actions.append(action.squeeze(0))  # ()
            episode_logprobs.append(log_prob.squeeze(0))  # ()
            episode_values.append(value_step.squeeze(0))  # ()

            obs = next_obs

        final_reward = float(reward)

        for _ in episode_obs:
            returns_buf.append(final_reward)

        obs_buf.extend(episode_obs)
        actions_buf.extend(episode_actions)
        logprobs_buf.extend(episode_logprobs)
        values_buf.extend(episode_values)

    obs_tensor = torch.stack(obs_buf, dim=0).to(device)         # (N, T)
    actions_tensor = torch.stack(actions_buf, dim=0).to(device) # (N,)
    logprobs_old_tensor = torch.stack(logprobs_buf, dim=0).to(device)  # (N,)
    values_tensor = torch.stack(values_buf, dim=0).to(device)   # (N,)
    returns_tensor = torch.tensor(returns_buf, dtype=torch.float32, device=device)  # (N,)

    # advantages = returns - values
    advantages_tensor = returns_tensor - values_tensor
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std() + 1e-8
    )

    return {
        "obs": obs_tensor,
        "actions": actions_tensor,
        "logprobs_old": logprobs_old_tensor,
        "values_old": values_tensor,
        "returns": returns_tensor,
        "advantages": advantages_tensor,
    }


def ppo_train(
    actor_critic: SmilesActorCritic,
    env: SmilesRLEnv,
    num_updates: int = 200,
    episodes_per_update: int = 16,
    ppo_epochs: int = 4,
    mini_batch_size: int = 64,
    lr: float = 3e-4,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_len: int = 120,
    temperature: float = 1.0,
    log_every: int = 10,
    eval_every: int = 20,
    eval_num_samples: int = 256,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> SmilesActorCritic:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic.to(device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    for update_idx in range(1, num_updates + 1):
        rollout = _collect_rollout(
            actor_critic=actor_critic,
            env=env,
            episodes_per_update=episodes_per_update,
            max_len=max_len,
            device=device,
            temperature=temperature,
        )

        obs = rollout["obs"]           # (N, T)
        actions = rollout["actions"]   # (N,)
        logprobs_old = rollout["logprobs_old"]  # (N,)
        returns = rollout["returns"]   # (N,)
        advantages = rollout["advantages"]  # (N,)

        N = obs.size(0)
        idxs = np.arange(N)

        actor_critic.train()
        for epoch in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = idxs[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_obs = obs[mb_idx]           # (B, T)
                mb_actions = actions[mb_idx]   # (B,)
                mb_logprobs_old = logprobs_old[mb_idx]  # (B,)
                mb_returns = returns[mb_idx]   # (B,)
                mb_advantages = advantages[mb_idx]  # (B,)

                logits, values = actor_critic(mb_obs)  # logits: (B, T, V), values: (B, T)

                with torch.no_grad():
                    mask = (mb_obs != 0)  # (B, T)
                    last_idx = mask.sum(dim=1) - 1       # (B,)
                B = mb_obs.size(0)

                flat_logits = logits.view(B * logits.size(1), logits.size(2))  # (B*T, V)
                flat_values = values.view(B * values.size(1))                  # (B*T,)

                offsets = torch.arange(B, device=device) * logits.size(1)
                idx_flat = offsets + last_idx
                logits_step = flat_logits[idx_flat]    # (B, V)
                values_step = flat_values[idx_flat]    # (B,)

                dist = Categorical(logits=logits_step)
                logprobs = dist.log_prob(mb_actions)   # (B,)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratios = torch.exp(logprobs - mb_logprobs_old)  # (B,)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_step, mb_returns)

                loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=5.0)
                optimizer.step()

        if log_every > 0 and update_idx % log_every == 0:
            avg_return = float(returns.mean().item())
            print(
                f"[PPO Update {update_idx}/{num_updates}] "
                f"Avg episode reward (QED): {avg_return:.4f}"
            )

        if eval_every > 0 and update_idx % eval_every == 0:
            metrics = evaluate_policy_ppo(
                actor_critic,
                vocab=env.vocab,
                num_samples=eval_num_samples,
                max_len=max_len,
                device=device,
            )
            print(
                f"  [Eval @ Update {update_idx}] "
                f"Avg QED: {metrics['avg_qed']:.4f}, "
                f"Validity: {metrics['validity'] * 100:.2f}%, "
                f"Novelty: {metrics['novelty'] * 100:.2f}%"
            )
            if save_path is not None:
                torch.save(actor_critic.state_dict(), save_path)
                print(f"  Saved PPO actor-critic to {save_path}")

    return actor_critic


@torch.no_grad()
def evaluate_policy_ppo(
    actor_critic: SmilesActorCritic,
    vocab: SmilesVocab,
    num_samples: int = 256,
    max_len: int = 120,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:

    if device is None:
        device = next(actor_critic.parameters()).device

    actor_critic.eval()

    smiles_list: List[str] = []
    qeds: List[float] = []
    valid_flags: List[bool] = []

    for _ in range(num_samples):
        s = actor_critic.sample(
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
