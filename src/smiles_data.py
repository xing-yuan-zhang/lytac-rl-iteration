import os
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SmilesVocab:

    PAD = "<PAD>"
    START = "<START>"
    END = "<END>"
    UNK = "<UNK>"

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(tokens)}

        self.pad_id = self.token_to_id[self.PAD]
        self.start_id = self.token_to_id[self.START]
        self.end_id = self.token_to_id[self.END]
        self.unk_id = self.token_to_id[self.UNK]

    @classmethod
    def build_from_smiles(
        cls,
        smiles_list: List[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> "SmilesVocab":
        counter = Counter()
        for s in smiles_list:
            counter.update(list(s.strip()))

        tokens_and_freq = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        base_tokens = [t for t, f in tokens_and_freq if f >= min_freq]
        if max_size is not None:
            base_tokens = base_tokens[: max(0, max_size - 4)]  # -4 for <PAD>, <START>, <END>, <UNK>

        special_tokens = [cls.PAD, cls.START, cls.END, cls.UNK]
        tokens = special_tokens + base_tokens
        return cls(tokens)

    def encode(self, smiles: str, add_start_end: bool = True) -> List[int]:
        chars = list(smiles.strip())
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in chars]
        if add_start_end:
            ids = [self.start_id] + ids + [self.end_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            t = self.id_to_token.get(i, self.UNK)
            if skip_special and t in {self.PAD, self.START, self.END}:
                continue
            tokens.append(t)
        return "".join(tokens)

    def __len__(self) -> int:
        return len(self.tokens)


class SmilesDataset(Dataset):

    def __init__(
        self,
        smiles_list: List[str],
        vocab: SmilesVocab,
        max_len: int = 120,
    ):
        self.vocab = vocab
        self.max_len = max_len

        self.smiles_list: List[str] = []
        for s in smiles_list:
            if len(s.strip()) + 2 <= max_len:
                self.smiles_list.append(s.strip())
            else:
                self.smiles_list.append(s.strip()[: max_len - 2])

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self.smiles_list[idx]
        token_ids = self.vocab.encode(smiles, add_start_end=True)

        # input:  <START> s1 ... sN
        # target: s1     ... sN <END>
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def smiles_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_seqs = [item["input_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    max_len = max(seq.size(0) for seq in input_seqs)
    pad_id = 0

    if hasattr(batch[0]["input_ids"], "pad_id"):
        pad_id = batch[0]["input_ids"].pad_id

    padded_inputs = []
    padded_targets = []
    input_masks = []

    for inp, tgt in zip(input_seqs, target_seqs):  # Batch collate
        pad_len = max_len - inp.size(0)
        padded_inputs.append(
            torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        padded_targets.append(
            torch.cat([tgt, torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        input_masks.append(
            torch.cat(
                [
                    torch.ones_like(inp, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ]
            )
        )

    return {
        "input_ids": torch.stack(padded_inputs, dim=0),
        "target_ids": torch.stack(padded_targets, dim=0),
        "attention_mask": torch.stack(input_masks, dim=0)
    }


def load_smiles_from_file(path: str) -> List[str]:

    if not os.path.isfile(path):
        raise FileNotFoundError(f"SMILES file not found: {path}")

    smiles_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            smiles = line.split()[0]
            smiles_list.append(smiles)
    return smiles_list


def create_smiles_dataloader(
    smiles_path: str,
    batch_size: int = 64,
    max_len: int = 120,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, SmilesVocab]:

    smiles_list = load_smiles_from_file(smiles_path)
    vocab = SmilesVocab.build_from_smiles(
        smiles_list,
        min_freq=min_freq,
        max_size=max_vocab_size,
    )

    assert vocab.pad_id == 0, "PAD token must have id 0 for current collate_fn."

    dataset = SmilesDataset(smiles_list, vocab, max_len=max_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=smiles_collate_fn,
    )

    return dataloader, vocab
