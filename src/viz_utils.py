from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import QED

from eval_utils import compute_metrics_from_smiles


def compute_qeds_and_validity(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    qeds = []
    valids = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            qeds.append(0.0)
            valids.append(False)
        else:
            qeds.append(float(QED.qed(mol)))
            valids.append(True)

    return np.array(qeds, dtype=float), np.array(valids, dtype=bool)


def plot_qed_histograms(
    model_smiles_dict: Dict[str, List[str]],
    bins: int = 30,
    save_path: Optional[str] = None,
    xlim: Tuple[float, float] = (0.0, 1.0),
) -> None:
    plt.figure(figsize=(8, 5))

    for model_name, smiles_list in model_smiles_dict.items():
        qeds, _ = compute_qeds_and_validity(smiles_list)
        qeds_valid = qeds[qeds > 0.0]
        if len(qeds_valid) == 0:
            continue

        plt.hist(
            qeds_valid,
            bins=bins,
            range=xlim,
            alpha=0.5,
            label=model_name,
            density=True,
        )

    plt.xlabel("QED")
    plt.ylabel("Density")
    plt.title("QED Distributions of Different Models")
    plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[viz_utils] Saved QED histogram to {save_path}")
    else:
        plt.show()


def plot_metric_bars(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:

    model_names = list(metrics_dict.keys())
    avg_qeds = [metrics_dict[m].get("avg_qed", 0.0) for m in model_names]
    validities = [metrics_dict[m].get("validity", 0.0) * 100 for m in model_names]
    novelties = [metrics_dict[m].get("novelty", 0.0) * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(8, 5))

    plt.bar(x - width, avg_qeds, width, label="Avg QED")
    plt.bar(x, validities, width, label="Validity (%)")
    plt.bar(x + width, novelties, width, label="Novelty (%)")

    plt.xticks(x, model_names, rotation=30, ha="right")
    plt.ylabel("Metric value")
    plt.title("Model Comparison: QED, Validity, Novelty")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[viz_utils] Saved metric bar plot to {save_path}")
    else:
        plt.show()


def compute_and_print_metrics(
    model_smiles_dict: Dict[str, List[str]],
    train_smiles_set: Optional[set] = None,
) -> Dict[str, Dict[str, float]]:
    from pprint import pprint

    metrics_dict: Dict[str, Dict[str, float]] = {}

    for model_name, smiles_list in model_smiles_dict.items():
        metrics = compute_metrics_from_smiles(
            smiles_list,
            train_smiles_set=train_smiles_set,
        )
        metrics_dict[model_name] = metrics

    print("[viz_utils] Metrics summary:")
    pprint(metrics_dict)
    return metrics_dict


def draw_top_molecules_grid(
    smiles_list,
    qeds=None,
    n_cols=3,
    sub_img_size=(300, 300),
    save_path="figures/top9_qed_molecules.png",
):
    mols = []
    legends = []

    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        mols.append(mol)
        if qeds is not None and i < len(qeds):
            legends.append(f"QED={qeds[i]:.2f}")
        else:
            legends.append("")

    if len(mols) == 0:
        print("[draw_top_molecules_grid] No valid molecules to draw.")
        return

    n_rows = (len(mols) + n_cols - 1) // n_cols

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=False,
    )

    img.save(save_path)
    print(f"[draw_top_molecules_grid] Saved grid image to {save_path}")
