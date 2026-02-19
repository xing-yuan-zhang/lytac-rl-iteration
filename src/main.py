import os
import sys
import argparse
import torch

def _add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)

def cmd_pretrain(a):
    _add_src_to_path()
    from smiles_data import create_smiles_dataloader
    from smiles_model import pretrain_smiles_rnn, sample_smiles_batch, quick_validity_check

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    train_loader, vocab = create_smiles_dataloader(
        a.data,
        batch_size=a.batch_size,
        max_len=a.max_len,
        min_freq=a.min_freq,
        max_vocab_size=None if a.max_vocab_size <= 0 else a.max_vocab_size,
    )

    os.makedirs(os.path.dirname(a.save), exist_ok=True) if os.path.dirname(a.save) else None

    model = pretrain_smiles_rnn(
        train_loader=train_loader,
        vocab=vocab,
        num_epochs=a.epochs,
        lr=a.lr,
        device=device,
        log_every=a.log_every,
        save_path=a.save,
    )

    samples = sample_smiles_batch(
        model,
        vocab,
        num_samples=a.num_samples,
        max_len=a.max_len,
        greedy=False,
        device=device,
    )
    print("Sampled SMILES:")
    for s in samples:
        print(s)

    v = quick_validity_check(
        model,
        vocab,
        num_samples=a.validity_samples,
        max_len=a.max_len,
        device=device,
    )
    print(f"Validity over {a.validity_samples} samples: {v * 100:.2f}%")

def cmd_reinforce(a):
    _add_src_to_path()
    from smiles_data import create_smiles_dataloader
    from smiles_model import SmilesRNNPolicy
    from smiles_rl_env import SmilesRLEnv
    from smiles_reinforce import reinforce_train, evaluate_policy

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    _, vocab = create_smiles_dataloader(
        a.data,
        batch_size=a.batch_size,
        max_len=a.max_len,
        min_freq=a.min_freq,
        max_vocab_size=None if a.max_vocab_size <= 0 else a.max_vocab_size,
    )

    policy = SmilesRNNPolicy(
        vocab_size=len(vocab),
        embed_dim=a.embed_dim,
        hidden_dim=a.hidden_dim,
        num_layers=a.num_layers,
        dropout=a.dropout,
    ).to(device)

    if a.pretrained:
        try:
            policy.load_state_dict(torch.load(a.pretrained, map_location=device))
            print(f"Loaded pretrained weights from {a.pretrained}")
        except FileNotFoundError:
            print("Warning: pretrained checkpoint not found, starting RL from scratch.")

    env = SmilesRLEnv(
        vocab=vocab,
        max_len=a.max_len,
        invalid_reward=a.invalid_reward,
        qed_scale=a.qed_scale,
    )

    os.makedirs(os.path.dirname(a.save), exist_ok=True) if os.path.dirname(a.save) else None

    policy = reinforce_train(
        policy=policy,
        env=env,
        num_episodes=a.num_episodes,
        batch_episodes=a.batch_episodes,
        lr=a.lr,
        reward_scale=a.reward_scale,
        baseline_momentum=a.baseline_momentum,
        log_every=a.log_every,
        eval_every=a.eval_every,
        eval_num_samples=a.eval_num_samples,
        max_len=a.max_len,
        device=device,
        save_path=a.save,
    )

    metrics = evaluate_policy(
        policy=policy,
        vocab=vocab,
        num_samples=a.final_eval_samples,
        max_len=a.max_len,
        device=device,
    )
    print("Final policy metrics:")
    print(metrics)

def cmd_ppo(a):
    _add_src_to_path()
    from smiles_data import create_smiles_dataloader
    from smiles_model import SmilesRNNPolicy
    from smiles_rl_env import SmilesRLEnv
    from smiles_PPO import SmilesActorCritic, ppo_train, evaluate_policy_ppo

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    _, vocab = create_smiles_dataloader(
        a.data,
        batch_size=a.batch_size,
        max_len=a.max_len,
        min_freq=a.min_freq,
        max_vocab_size=None if a.max_vocab_size <= 0 else a.max_vocab_size,
    )

    policy = SmilesRNNPolicy(
        vocab_size=len(vocab),
        embed_dim=a.embed_dim,
        hidden_dim=a.hidden_dim,
        num_layers=a.num_layers,
        dropout=a.dropout,
    ).to(device)

    if a.pretrained:
        try:
            policy.load_state_dict(torch.load(a.pretrained, map_location=device))
            print(f"Loaded pretrained weights from {a.pretrained}")
        except FileNotFoundError:
            print("Warning: pretrained checkpoint not found, PPO will start from random policy.")

    actor_critic = SmilesActorCritic.from_pretrained_policy(policy, dropout=a.dropout).to(device)

    env = SmilesRLEnv(
        vocab=vocab,
        max_len=a.max_len,
        invalid_reward=a.invalid_reward,
        qed_scale=a.qed_scale,
    )

    os.makedirs(os.path.dirname(a.save), exist_ok=True) if os.path.dirname(a.save) else None

    actor_critic = ppo_train(
        actor_critic=actor_critic,
        env=env,
        num_updates=a.num_updates,
        episodes_per_update=a.episodes_per_update,
        ppo_epochs=a.ppo_epochs,
        mini_batch_size=a.mini_batch_size,
        lr=a.lr,
        clip_range=a.clip_range,
        value_coef=a.value_coef,
        entropy_coef=a.entropy_coef,
        max_len=a.max_len,
        temperature=a.temperature,
        log_every=a.log_every,
        eval_every=a.eval_every,
        eval_num_samples=a.eval_num_samples,
        device=device,
        save_path=a.save,
    )

    metrics = evaluate_policy_ppo(
        actor_critic,
        vocab=vocab,
        num_samples=a.final_eval_samples,
        max_len=a.max_len,
        device=device,
    )
    print("Final PPO policy metrics:")
    print(metrics)

def cmd_eval(a):
    _add_src_to_path()
    from smiles_data import create_smiles_dataloader
    from smiles_model import SmilesRNNPolicy
    from smiles_PPO import SmilesActorCritic
    from eval_utils import (
        load_train_smiles_set,
        smiles_list_to_set,
        sample_random_from_dataset,
        sample_from_rnn_policy,
        sample_from_actor_critic,
        compute_metrics_from_smiles,
        compute_dataset_qed_stats,
        get_top_k_novel_molecules_unique,
        print_metrics_table,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    train_smiles_list = load_train_smiles_set(a.data, max_size=a.train_subset)
    train_smiles_set = smiles_list_to_set(train_smiles_list)
    print(f"Train SMILES subset size: {len(train_smiles_list)}, canonical set size: {len(train_smiles_set)}")

    _, vocab = create_smiles_dataloader(a.data, batch_size=a.batch_size, max_len=a.max_len)

    pretrained = SmilesRNNPolicy(
        vocab_size=len(vocab), embed_dim=a.embed_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout
    ).to(device)
    pretrained.load_state_dict(torch.load(a.pretrained_ckpt, map_location=device))
    print(f"Loaded pretrained RNN policy from {a.pretrained_ckpt}")

    reinforce = SmilesRNNPolicy(
        vocab_size=len(vocab), embed_dim=a.embed_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout
    ).to(device)
    reinforce.load_state_dict(torch.load(a.reinforce_ckpt, map_location=device))
    print(f"Loaded REINFORCE-tuned policy from {a.reinforce_ckpt}")

    actor_critic = SmilesActorCritic(
        vocab_size=len(vocab), embed_dim=a.embed_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout
    ).to(device)
    actor_critic.load_state_dict(torch.load(a.ppo_ckpt, map_location=device))
    print(f"Loaded PPO actor-critic from {a.ppo_ckpt}")

    n = a.num_eval_samples

    dataset_samples = sample_random_from_dataset(train_smiles_list, num_samples=n)
    m_dataset = compute_metrics_from_smiles(dataset_samples, train_smiles_set=train_smiles_set)

    s_pre = sample_from_rnn_policy(pretrained, vocab=vocab, num_samples=n, max_len=a.max_len, greedy=False, device=device)
    m_pre = compute_metrics_from_smiles(s_pre, train_smiles_set=train_smiles_set)

    s_re = sample_from_rnn_policy(reinforce, vocab=vocab, num_samples=n, max_len=a.max_len, greedy=False, device=device)
    m_re = compute_metrics_from_smiles(s_re, train_smiles_set=train_smiles_set)

    s_ppo = sample_from_actor_critic(actor_critic, vocab=vocab, num_samples=n, max_len=a.max_len, greedy=False, device=device)
    m_ppo = compute_metrics_from_smiles(s_ppo, train_smiles_set=train_smiles_set)

    results = {"dataset_random": m_dataset, "pretrained_rnn": m_pre, "reinforce": m_re, "ppo": m_ppo}
    print("\n=== Model Comparison ===")
    print_metrics_table(results)

    if a.dataset_qed_subset > 0:
        ds = compute_dataset_qed_stats(train_smiles_list, max_eval_size=a.dataset_qed_subset)
        print("\nDataset QED stats (subset):", ds)

    if a.top_k > 0:
        top = get_top_k_novel_molecules_unique(smiles_list=s_ppo, train_smiles_set=train_smiles_set, k=a.top_k)
        print(f"\n=== Top-{a.top_k} High-QED Novel Molecules from PPO ===")
        for i, r in enumerate(top, start=1):
            print(f"[{i:02d}] QED={r['qed']:.4f} | SMILES={r['smiles']}")

def cmd_viz(a):
    _add_src_to_path()
    from eval_utils import load_train_smiles_set, smiles_list_to_set
    from smiles_data import create_smiles_dataloader
    from smiles_model import SmilesRNNPolicy
    from smiles_PPO import SmilesActorCritic
    from viz_utils import (
        plot_qed_histograms,
        plot_metric_bars,
        compute_and_print_metrics,
        draw_top_molecules_grid,
        get_top_k_novel_molecules,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    train_smiles_list = load_train_smiles_set(a.data, max_size=a.train_subset)
    train_smiles_set = smiles_list_to_set(train_smiles_list)
    _, vocab = create_smiles_dataloader(a.data, batch_size=a.batch_size, max_len=a.max_len)

    if a.model_type == "rnn":
        m = SmilesRNNPolicy(
            vocab_size=len(vocab), embed_dim=a.embed_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout
        ).to(device)
        m.load_state_dict(torch.load(a.ckpt, map_location=device))
        samples = compute_and_print_metrics(m, vocab, train_smiles_set, device=device, max_len=a.max_len, num_samples=a.num_samples)
        top = get_top_k_novel_molecules(samples, train_smiles_set, k=a.top_k)
    else:
        m = SmilesActorCritic(
            vocab_size=len(vocab), embed_dim=a.embed_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout
        ).to(device)
        m.load_state_dict(torch.load(a.ckpt, map_location=device))
        samples = compute_and_print_metrics(m, vocab, train_smiles_set, device=device, max_len=a.max_len, num_samples=a.num_samples, is_actor_critic=True)
        top = get_top_k_novel_molecules(samples, train_smiles_set, k=a.top_k)

    plot_qed_histograms(samples, train_smiles_set, title_prefix=a.title)
    plot_metric_bars(samples, train_smiles_set, title_prefix=a.title)
    if a.draw_grid and top:
        draw_top_molecules_grid(top, title=a.title)

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")

    sp = p.add_subparsers(dest="cmd", required=True)

    p0 = sp.add_parser("pretrain")
    p0.add_argument("--data", default="data/guacamol_v1_all.smiles")
    p0.add_argument("--batch_size", type=int, default=128)
    p0.add_argument("--max_len", type=int, default=120)
    p0.add_argument("--min_freq", type=int, default=1)
    p0.add_argument("--max_vocab_size", type=int, default=-1)
    p0.add_argument("--epochs", type=int, default=2)
    p0.add_argument("--lr", type=float, default=5e-4)
    p0.add_argument("--log_every", type=int, default=100)
    p0.add_argument("--save", default="checkpoints/smiles_rnn_pretrained.pt")
    p0.add_argument("--num_samples", type=int, default=10)
    p0.add_argument("--validity_samples", type=int, default=500)
    p0.set_defaults(func=cmd_pretrain)

    pr = sp.add_parser("reinforce")
    pr.add_argument("--data", default="data/guacamol_v1_all.smiles")
    pr.add_argument("--batch_size", type=int, default=128)
    pr.add_argument("--max_len", type=int, default=120)
    pr.add_argument("--min_freq", type=int, default=1)
    pr.add_argument("--max_vocab_size", type=int, default=-1)
    pr.add_argument("--embed_dim", type=int, default=256)
    pr.add_argument("--hidden_dim", type=int, default=512)
    pr.add_argument("--num_layers", type=int, default=2)
    pr.add_argument("--dropout", type=float, default=0.1)
    pr.add_argument("--pretrained", default="checkpoints/smiles_rnn_pretrained.pt")
    pr.add_argument("--invalid_reward", type=float, default=0.0)
    pr.add_argument("--qed_scale", default="none")
    pr.add_argument("--num_episodes", type=int, default=200)
    pr.add_argument("--batch_episodes", type=int, default=16)
    pr.add_argument("--lr", type=float, default=1e-4)
    pr.add_argument("--reward_scale", type=float, default=5.0)
    pr.add_argument("--baseline_momentum", type=float, default=0.9)
    pr.add_argument("--log_every", type=int, default=10)
    pr.add_argument("--eval_every", type=int, default=100)
    pr.add_argument("--eval_num_samples", type=int, default=256)
    pr.add_argument("--final_eval_samples", type=int, default=512)
    pr.add_argument("--save", default="checkpoints/smiles_policy_reinforce.pt")
    pr.set_defaults(func=cmd_reinforce)

    pp = sp.add_parser("ppo")
    pp.add_argument("--data", default="data/guacamol_v1_all.smiles")
    pp.add_argument("--batch_size", type=int, default=128)
    pp.add_argument("--max_len", type=int, default=120)
    pp.add_argument("--min_freq", type=int, default=1)
    pp.add_argument("--max_vocab_size", type=int, default=-1)
    pp.add_argument("--embed_dim", type=int, default=256)
    pp.add_argument("--hidden_dim", type=int, default=512)
    pp.add_argument("--num_layers", type=int, default=2)
    pp.add_argument("--dropout", type=float, default=0.1)
    pp.add_argument("--pretrained", default="checkpoints/smiles_rnn_pretrained.pt")
    pp.add_argument("--invalid_reward", type=float, default=0.0)
    pp.add_argument("--qed_scale", default="none")
    pp.add_argument("--num_updates", type=int, default=200)
    pp.add_argument("--episodes_per_update", type=int, default=16)
    pp.add_argument("--ppo_epochs", type=int, default=4)
    pp.add_argument("--mini_batch_size", type=int, default=64)
    pp.add_argument("--lr", type=float, default=3e-4)
    pp.add_argument("--clip_range", type=float, default=0.2)
    pp.add_argument("--value_coef", type=float, default=0.5)
    pp.add_argument("--entropy_coef", type=float, default=0.01)
    pp.add_argument("--temperature", type=float, default=1.0)
    pp.add_argument("--log_every", type=int, default=5)
    pp.add_argument("--eval_every", type=int, default=20)
    pp.add_argument("--eval_num_samples", type=int, default=256)
    pp.add_argument("--final_eval_samples", type=int, default=512)
    pp.add_argument("--save", default="checkpoints/smiles_actor_critic_ppo.pt")
    pp.set_defaults(func=cmd_ppo)

    pe = sp.add_parser("eval")
    pe.add_argument("--data", default="data/guacamol_v1_all.smiles")
    pe.add_argument("--batch_size", type=int, default=128)
    pe.add_argument("--max_len", type=int, default=120)
    pe.add_argument("--train_subset", type=int, default=200000)
    pe.add_argument("--num_eval_samples", type=int, default=512)
    pe.add_argument("--dataset_qed_subset", type=int, default=50000)
    pe.add_argument("--top_k", type=int, default=20)
    pe.add_argument("--embed_dim", type=int, default=256)
    pe.add_argument("--hidden_dim", type=int, default=512)
    pe.add_argument("--num_layers", type=int, default=2)
    pe.add_argument("--dropout", type=float, default=0.1)
    pe.add_argument("--pretrained_ckpt", default="checkpoints/smiles_rnn_pretrained.pt")
    pe.add_argument("--reinforce_ckpt", default="checkpoints/smiles_policy_reinforce.pt")
    pe.add_argument("--ppo_ckpt", default="checkpoints/smiles_actor_critic_ppo.pt")
    pe.set_defaults(func=cmd_eval)

    pv = sp.add_parser("viz")
    pv.add_argument("--data", default="data/guacamol_v1_all.smiles")
    pv.add_argument("--batch_size", type=int, default=128)
    pv.add_argument("--max_len", type=int, default=120)
    pv.add_argument("--train_subset", type=int, default=200000)
    pv.add_argument("--num_samples", type=int, default=512)
    pv.add_argument("--top_k", type=int, default=20)
    pv.add_argument("--model_type", choices=["rnn", "ppo"], default="ppo")
    pv.add_argument("--ckpt", default="checkpoints/smiles_actor_critic_ppo.pt")
    pv.add_argument("--title", default="")
    pv.add_argument("--draw_grid", action="store_true")
    pv.add_argument("--embed_dim", type=int, default=256)
    pv.add_argument("--hidden_dim", type=int, default=512)
    pv.add_argument("--num_layers", type=int, default=2)
    pv.add_argument("--dropout", type=float, default=0.1)
    pv.set_defaults(func=cmd_viz)

    return p

def main():
    p = build_parser()
    a = p.parse_args()
    a.func(a)

if __name__ == "__main__":
    main()
