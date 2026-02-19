from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import QED

from smiles_data import SmilesVocab, load_smiles_from_file
from smiles_model import SmilesRNNPolicy
from smiles_PPO import SmilesActorCritic


def load_train_smiles_set(
    path: str,
    max_size: Optional[int] = None,
) -> List[str]:

    smiles_list = load_smiles_from_file(path)
    if max_size is not None and len(smiles_list) > max_size:
        idx = np.random.choice(len(smiles_list), size=max_size, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
    return smiles_list


def smiles_list_to_set(smiles_list: List[str]) -> set:
    canon_set = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        canon_set.add(canon)
    return canon_set


def compute_metrics_from_smiles(
    smiles_list: List[str],
    train_smiles_set: Optional[set] = None,
) -> Dict[str, float]:
    qeds: List[float] = []
    valid_flags: List[bool] = []
    canon_list: List[str] = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            valid_flags.append(False)
            qeds.append(0.0)
            canon_list.append("")
        else:
            valid_flags.append(True)
            qed = float(QED.qed(mol))
            qeds.append(qed)
            canon = Chem.MolToSmiles(mol, canonical=True)
            canon_list.append(canon)

    avg_qed = float(np.mean(qeds)) if qeds else 0.0
    validity = float(np.mean(valid_flags)) if valid_flags else 0.0

    if train_smiles_set is not None:
        novel_flags = [
            (c != "" and c not in train_smiles_set) for c in canon_list
        ]
        novelty = float(np.mean(novel_flags)) if novel_flags else 0.0
    else:
        unique_canon = set([c for c in canon_list if c != ""])
        total_valid = sum(1 for c in canon_list if c != "")
        novelty = len(unique_canon) / max(1, total_valid)

    return {
        "avg_qed": avg_qed,
        "validity": validity,
        "novelty": novelty,
    }


def get_top_k_novel_molecules_unique(
    smiles_list: List[str],
    train_smiles_set: Optional[set] = None,
    k: int = 20,
) -> List[Dict[str, object]]:
    records = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue

        qed = float(QED.qed(mol))
        canon = Chem.MolToSmiles(mol, canonical=True)
        is_novel = (train_smiles_set is None) or (canon not in train_smiles_set)

        records.append(
            {
                "smiles": s,
                "canon_smiles": canon,
                "qed": qed,
                "is_valid": True,
                "is_novel": is_novel,
            }
        )

    records = [r for r in records if r["is_valid"] and r["is_novel"]]

    records.sort(key=lambda r: r["qed"], reverse=True)

    unique_by_canon = {}
    for r in records:
        c = r["canon_smiles"]
        if c not in unique_by_canon:
            unique_by_canon[c] = r
        if len(unique_by_canon) >= k:
            break

    return list(unique_by_canon.values())


@torch.no_grad()
def sample_from_rnn_policy(
    policy: SmilesRNNPolicy,
    vocab: SmilesVocab,
    num_samples: int = 256,
    max_len: int = 120,
    greedy: bool = False,
    device: Optional[torch.device] = None,
) -> List[str]:
    if device is None:
        device = next(policy.parameters()).device

    policy.eval()
    smiles_list: List[str] = []
    for _ in range(num_samples):
        s = policy.sample(
            vocab=vocab,
            max_len=max_len,
            greedy=greedy,
            device=device,
        )
        smiles_list.append(s)
    return smiles_list


@torch.no_grad()
def sample_from_actor_critic(
    actor_critic: SmilesActorCritic,
    vocab: SmilesVocab,
    num_samples: int = 256,
    max_len: int = 120,
    greedy: bool = False,
    device: Optional[torch.device] = None,
) -> List[str]:
    if device is None:
        device = next(actor_critic.parameters()).device

    actor_critic.eval()
    smiles_list: List[str] = []
    for _ in range(num_samples):
        s = actor_critic.sample(
            vocab=vocab,
            max_len=max_len,
            greedy=greedy,
            device=device,
        )
        smiles_list.append(s)
    return smiles_list


def sample_random_from_dataset(
    train_smiles_list: List[str],
    num_samples: int = 256,
) -> List[str]:
    if len(train_smiles_list) == 0:
        return []
    idx = np.random.randint(0, len(train_smiles_list), size=num_samples)
    return [train_smiles_list[i] for i in idx]


def compute_dataset_qed_stats(
    train_smiles_list: List[str],
    max_eval_size: Optional[int] = 50000,
) -> Dict[str, float]:
    if max_eval_size is not None and len(train_smiles_list) > max_eval_size:
        idx = np.random.choice(len(train_smiles_list), size=max_eval_size, replace=False)
        eval_list = [train_smiles_list[i] for i in idx]
    else:
        eval_list = train_smiles_list

    qeds = []
    for s in eval_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        qeds.append(float(QED.qed(mol)))

    if not qeds:
        return {"avg_qed": 0.0, "top_qed": 0.0}

    return {
        "avg_qed": float(np.mean(qeds)),
        "top_qed": float(np.max(qeds)),
    }


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
) -> None:
    header = f"{'Model':<18} | {'Avg QED':>8} | {'Validity':>9} | {'Novelty':>9}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        avg_qed = m.get("avg_qed", 0.0)
        validity = m.get("validity", 0.0)
        novelty = m.get("novelty", 0.0)
        print(
            f"{name:<18} | "
            f"{avg_qed:8.4f} | "
            f"{validity*100:8.2f}% | "
            f"{novelty*100:8.2f}%"
        )
