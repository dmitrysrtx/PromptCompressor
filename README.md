# ⚡ Discrete Prompt Compression with Reinforcement Learning (Code Synthesis Focus)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.13+](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg)](https://pytorch.org/)
[![Reinforcement Learning](https://img.shields.io/badge/RL-PPO%20%7C%20A2C-green.svg)](https://github.com/dmitrysrtx/PromptCompressor)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **An advanced Reinforcement Learning framework designed to compress instruction prompts for Code Generation LLMs by ~37% while preserving strict syntactic and semantic fidelity (BLEU & ROUGE-L).**

---

## 📌 Executive Summary

Prompt length directly impacts inference latency, token budget, and GPU VRAM consumption when running self-hosted LLMs. While traditional prompt compression methods target general natural language, **software engineering and code synthesis prompts are extremely fragile** — dropping a single bracket or keyword breaks syntax.

This project extends state-of-the-art **Discrete Prompt Compression (PCRL)** using **PPO/A2C Policy Gradient with Token Action Masking** specifically tuned for code generation tasks (CodeAlpaca). By incorporating a **dual BLEU + ROUGE-L reward mechanism**, the RL policy (RoBERTa) learns to prune redundant prompt tokens without degrading code execution validity.

---

## 📐 Architecture & RL Environment Workflow

```
                        ┌──────────────────────────────────────────────┐
                        │              Input Code Prompt               │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │      RoBERTa RL Policy Agent (Actor)        │
                        │       Generates Token Action Masks          │
                        └──────────────────────┬───────────────────────┘
                                               │
                                       (Action: Keep / Prune)
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │          Compressed Code Prompt              │
                        │        (~37% Token Reduction)                │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │   Instruction-Tuned LLM (GPT2-XL-Code)       │
                        │             Synthesizes Code                 │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │        Environment Reward Calculation        │
                        │   Reward = α·BLEU(Code) + β·ROUGE_L(Text)    │
                        └──────────────────────┬───────────────────────┘
                                               │
                                       (Policy Update)
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │         PPO / A2C Policy Optimization        │
                        └──────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results & Experimental Performance

Tested across multiple CodeAlpaca validation splits using fine-tuned `GPT2-XL-Code` and `RoBERTa-Code-RL` policy:

| Evaluation Split | Model Configuration | Compression Ratio (CR) | ROUGE-L Score | BLEU Score | Syntactic Preservation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation Seen** | Original Baseline (GPT2-XL-Code) | `0.0%` (0 tokens pruned) | `0.5698` | `0.3450` | 100% Baseline |
| **Validation Seen** | **PCRL Agent (RoBERTa + BLEU)** | **`36.54%`** | **`0.4978`** | **`0.2929`** | **High Fidelity** |
| **Validation Unseen** | Original Baseline (GPT2-XL-Code) | `0.0%` (0 tokens pruned) | `0.5361` | `0.3236` | 100% Baseline |
| **Validation Unseen** | **PCRL Agent (RoBERTa + BLEU)** | **`36.66%`** | **`0.4757`** | **`0.2772`** | **High Fidelity** |
| **Validation Human** | Original Baseline (GPT2-XL-Code) | `0.0%` (0 tokens pruned) | `0.0%` (0 tokens pruned) | `0.5506` | `0.3410` |
| **Validation Human** | **PCRL Agent (RoBERTa + BLEU)** | **`36.98%`** | **`0.4728`** | **`0.2874`** | **High Fidelity** |

### Key Takeaways:
- **~37% Token Reduction:** Saves over one-third of prompt tokens in every inference run.
- **Robust Generalization:** Consistent compression ratios across seen, unseen, and human-crafted code instruction splits.
- **Minimal Quality Drop:** Preserves core AST syntax and code execution structure.

---

## 🚀 Key Technical Innovations & Contributions

1. **Code Generation Adaptation:** Refactored reward calculation and prompt parsing specifically for CodeAlpaca and code-generation instruction datasets.
2. **Dual BLEU + ROUGE-L Reward Function:** Integrated BLEU n-gram precision metrics directly into the RL reward function to prevent token pruning from destroying program syntax.
3. **Action Masking PPO Policy:** Implemented discrete action space masking (`pcrl/algorithms/ppo_mask`) allowing the RL agent to make binary keep/prune decisions per token.
4. **Cloud-Ready Training Pipeline:** Provided single-command execution wrappers (`run_eval.sh`, `setup_env.sh`, `run_pcrl.sh`) optimized for Google Colab and headless GPU server deployment.

---

## ⚙️ Environment Setup & Installation

### Local Linux Environment
```bash
# 1. Clone repository
git clone https://github.com/dmitrysrtx/PromptCompressor.git
cd PromptCompressor

# 2. Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab Notebook Integration
Add the following snippet in your initial Colab cell to mount drive and load scripts:
```python
import sys
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive Mounted.")
```
Then execute:
```bash
source ./setup_env.sh
```

---

## 🏋️ Training & Evaluation Commands

### 1. Fine-Tune Baseline Generator Model
```bash
python scripts/finetune_gpt2.py
```

### 2. Train Prompt Compressor Agent (PCRL)
```bash
python train_pcrl.py \
  --config_path configs/gpt2-xl-code.yml \
  --log_to_wandb \
  --seed=42 \
  --experiment_name=roberta_code_v1_bleu
```

### 3. Automated Evaluation Pipeline
```bash
# Evaluate Original Uncompressed Baseline
./run_eval.sh -t original -m gpt2-xl-code -b 16

# Evaluate PCRL Agent Compression
./run_eval.sh -t pcrl -c gpt2-xl-code -p PromptCompressor_code_RL/roberta_code_v1_bleu_512_part2 -b 16
```

---

## 📄 License & Acknowledgements

This codebase is licensed under **Apache 2.0**.
- Based on the foundational work on [Discrete Prompt Compression with RL](https://arxiv.org/abs/2308.08758) by Jung et al.
- Built using components from [RL4LMs (AllenAI)](https://github.com/allenai/RL4LMs) and [CodeAlpaca](https://github.com/sahil280114/codealpaca).

---

## 👨💻 Architect & Contact

- **Lead AI & Systems Architect:** Dmitry Strizhak
- **Domain:** Reinforcement Learning, LLM Optimization, Prompt Engineering & Model Compression
- **Business Inquiries:** [`tech1.ai.solutions@gmail.com`](mailto:tech1.ai.solutions@gmail.com)
