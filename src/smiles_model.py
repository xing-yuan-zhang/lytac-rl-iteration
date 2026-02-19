from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rdkit import Chem

from smiles_data import SmilesVocab


class SmilesRNNPolicy(nn.Module):
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
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        emb = self.embedding(input_ids)  # (B, T, E)
        output, hidden = self.lstm(emb, hidden)  # output: (B, T, H), hidden: (h_n, c_n)
        logits = self.output_head(output)  # (B, T, V)
        return logits, hidden

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
        hidden = None
        generated_ids: List[int] = [start_id]

        for _ in range(max_len - 1):
            logits, hidden = self.forward(input_ids, hidden)  # (1, T, V)
            logits_step = logits[:, -1, :]  # (1, V)

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

            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)

        smiles = vocab.decode(generated_ids, skip_special=True)
        return smiles


def pretrain_smiles_rnn(
    train_loader: DataLoader,
    vocab: SmilesVocab,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    log_every: int = 100,
    save_path: Optional[str] = None,
) -> SmilesRNNPolicy:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmilesRNNPolicy(vocab_size=len(vocab))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)      # (B, T)
            target_ids = batch["target_ids"].to(device)    # (B, T)
            # attention_mask = batch["attention_mask"].to(device)  # (B, T)

            optimizer.zero_grad()
            logits, _ = model(input_ids)  # (B, T, V)

            # reshape for CE: (B*T, V), (B*T)
            B, T, V = logits.size()
            loss = criterion(logits.view(B * T, V), target_ids.view(B * T))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if log_every > 0 and global_step % log_every == 0:
                print(
                    f"[Epoch {epoch} | Step {global_step}] "
                    f"Train loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"==> Epoch {epoch} finished. Avg loss: {avg_loss:.4f}")

        validity = quick_validity_check(model, vocab, num_samples=200, device=device)
        print(f"    Quick validity (200 samples): {validity * 100:.2f}%")

        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print(f"    Model checkpoint saved to {save_path}")

    return model


@torch.no_grad()
def sample_smiles_batch(
    model: SmilesRNNPolicy,
    vocab: SmilesVocab,
    num_samples: int = 100,
    max_len: int = 120,
    greedy: bool = False,
    device: Optional[torch.device] = None,
) -> List[str]:

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    smiles_list: List[str] = []
    for _ in range(num_samples):
        s = model.sample(vocab, max_len=max_len, greedy=greedy, device=device)
        smiles_list.append(s)
    return smiles_list


@torch.no_grad()
def quick_validity_check(
    model: SmilesRNNPolicy,
    vocab: SmilesVocab,
    num_samples: int = 200,
    max_len: int = 120,
    device: Optional[torch.device] = None,
) -> float:

    smiles_list = sample_smiles_batch(
        model, vocab, num_samples=num_samples, max_len=max_len, device=device
    )
    valid = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid += 1
    return valid / max(1, num_samples)
