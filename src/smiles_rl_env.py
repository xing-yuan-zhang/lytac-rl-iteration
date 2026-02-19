from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from rdkit import Chem
from rdkit.Chem import QED

from smiles_data import SmilesVocab


class SmilesRLEnv:
    def __init__(
        self,
        vocab: SmilesVocab,
        max_len: int = 120,
        invalid_reward: float = 0.0,
        qed_scale: str = "none",
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.invalid_reward = invalid_reward
        self.qed_scale = qed_scale

        self.start_id = vocab.start_id
        self.end_id = vocab.end_id
        self.pad_id = vocab.pad_id

        self.current_tokens: List[int] = []
        self.done: bool = False

    def reset(self) -> np.ndarray:
        self.current_tokens = [self.start_id]
        self.done = False
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        tokens = self.current_tokens.copy()
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]

        pad_len = self.max_len - len(tokens)
        tokens = tokens + [self.pad_id] * pad_len

        return np.array(tokens, dtype=np.int64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        self.current_tokens.append(int(action))

        if action == self.end_id or len(self.current_tokens) >= self.max_len:
            self.done = True
            reward, info = self._finalize_episode()
        else:
            reward = 0.0
            info = {}

        obs = self._get_obs()
        return obs, reward, self.done, info

    def _finalize_episode(self) -> Tuple[float, Dict[str, Any]]:
        smiles = self.vocab.decode(self.current_tokens, skip_special=True)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reward = float(self.invalid_reward)
            info = {
                "smiles": smiles,
                "is_valid": False,
                "qed": 0.0,
            }
            return reward, info

        qed = float(QED.qed(mol))
        if self.qed_scale == "log":
            reward = np.log(1e-8 + qed)  # optional nonlinearity
        else:
            reward = qed

        info = {
            "smiles": smiles,
            "is_valid": True,
            "qed": qed,
        }
        return reward, info

    def render(self) -> None:
        smiles = self.vocab.decode(self.current_tokens, skip_special=True)
        print(f"Current SMILES: {smiles}")

    @property
    def action_space_n(self) -> int:
        return len(self.vocab)  # number of possible actions = vocab size

    @property
    def observation_shape(self) -> Tuple[int]:
        return (self.max_len,)  # shape of observation array
