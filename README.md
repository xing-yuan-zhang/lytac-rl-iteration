# Policy optimization for drug like SMILES generation

This repository documents an exploratory project in our lab built on our submitted work *Cell-Selective and Organ-Specific Lysosomal Degradation of αv Integrins in Renal Fibrosis*. :contentReference[oaicite:0]{index=0}

The goal is to prototype an RL-assisted iteration loop for designing LYTAC–related small-molecule/linker motifs in SMILES space, where reinforcement learning nudges a generative prior toward more drug-like and constraint-satisfying candidates. :contentReference[oaicite:1]{index=1}

> Current implementation uses QED as a computationally cheap proxy reward to validate the workflow end-to-end, and the reward can be replaced/augmented with task-specific scores including filters, physicochemical windows, docking/ML predictors and linker constraints. :contentReference[oaicite:2]{index=2}

## Description

- **SMILES generation as an episodic MDP**: the agent emits tokens step-by-step until `<END>` or max length; reward is assigned at episode termination after RDKit parsing. :contentReference[oaicite:3]{index=3}  
- A **sequence model prior** (2-layer LSTM) trained on a drug-like corpus (GuacaMol/ChEMBL-derived) to learn a valid chemical distribution. :contentReference[oaicite:4]{index=4}  
- Two RL fine-tuning baselines:
  - **REINFORCE** with a moving-average baseline to reduce variance. :contentReference[oaicite:5]{index=5}  
  - **PPO (actor–critic)** with clipped objective and entropy regularization for exploration. :contentReference[oaicite:6]{index=6}  

This repo is intended as a drop-in optimization layer for a drug generation iteration loop:

- Replace pure QED reward with multi-objective reward:
  - drug-likeness + physicochemical windows (logP/PSA/HBD/HBA/rings)
  - synthetic accessibility / substructure filters (PAINS/reactive groups)
  - linker/warhead constraints (allowed motifs, attachment points, length)
  - optional medium-cost scoring (docking / ML proxy) and high-cost physics later :contentReference[oaicite:9]{index=9}  
- Add diversity control (scaffold or similarity penalties / diversity filters) to avoid collapse during PPO-style training. :contentReference[oaicite:10]{index=10}  

## Notes / disclaimer

This is a methods prototype. QED-optimized molecules are not guaranteed to be pharmaceutically active; any generated candidates must be screened with task-specific scoring and ultimately validated experimentally. :contentReference[oaicite:11]{index=11}  
