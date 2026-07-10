# Prompt Compression with Reinforcement Learning (Code Generation Focus)

## Introduction
This repository contains the code, datasets, and infrastructure for the study on "[Discrete Prompt Compression with Reinforcement Learning](https://arxiv.org/abs/2308.08758)". This specific fork and extension of the project aims to explore practical ways to compress highly fragile **software engineering and code generation prompts** in instruction-tuned models using Reinforcement Learning (RL). 

By acting as an intelligent semantic filter (utilizing a RoBERTa-based agent), the system reduces prompt token length, thereby minimizing VRAM consumption and inference latency for Local LLMs, while preserving the strict technical context required for accurate code synthesis.

## Key Features & Updates
- **Code Generation Focus:** Adapted the pipeline to evaluate technical prompt compression using the CodeAlpaca dataset.
- **BLEU Metric Integration:** Extended the core reward functions and evaluation algorithms to include the BLEU metric, which is critical for measuring the syntactic exactness of generated code alongside traditional ROUGE-L semantic overlap.
- **Google Colab Ready:** Includes automated shell scripts (`setup_env.sh`, `run_eval.sh`) for seamless execution and batch processing in cloud environments like Google Colab.

## Requirements

- Python 3.9 
- PyTorch 1.13.1+ (CUDA supported)
- Other dependencies listed in `requirements.txt`

## Setup

### Local Installation
1. Clone the repository:
```bash
   git clone [https://github.com/dmitrysrtx/PromptCompressor.git](https://github.com/dmitrysrtx/PromptCompressor.git)
   cd PromptCompressor
   ```
2. Install the required packages:
```bash
   pip install -r requirements.txt
   ```

### Google Colab Integration
If you are running the project inside a Google Colab environment, follow these steps to mount your storage and configure the virtual environment.

**Step 1:** Run the following Python code in the very first cell of your Jupyter Notebook to safely mount Google Drive:
```python
# Run this cell inside your Jupyter Notebook to mount Google Drive (Colab only)
import sys

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive successfully mounted!")
else:
    print("Running locally. Skipping Google Drive mount.")
```

**Step 2:** Navigate to your project directory inside the notebook using `%cd` and execute the environment setup script to handle configurations and virtual environments:
```bash
# Execute the environment setup script via terminal cell
source ./setup_env.sh
```

## Datasets
The primary dataset used in this study includes **CodeAlpaca**, alongside Alpaca+ from the [Mu et al. (2023) repository](https://github.com/jayelm/gisting). I have specifically utilized this data to conduct my experiments on prompt compression where rigid technical instructions are paramount. The data is located in `data/alpaca_plus` and relevant code instruction directories.

## Training
The training process consists of two main steps:

### 1. Instruction Fine-Tuning
First, we fine-tune the existing foundation models (e.g., `gpt2-xl-code`) on the target datasets. If you have a preferred instruction-tuned model, you can skip this section. 

```bash
python scripts/finetune_gpt2.py
```

### 2. Training PCRL (Prompt Compressor RL)
After fine-tuning the generative models, we train the RL policy (e.g., RoBERTa agent) utilizing the customized reward function incorporating BLEU and ROUGE metrics:

```bash
python train_pcrl.py --config_path configs/gpt2-xl-code.yml --log_to_wandb --seed=42 --experiment_name=roberta_code_v1_bleu
```

## Evaluation

The evaluation process calculates the **Compression Ratio (CR)** and the relative drop in **BLEU/ROUGE** metrics. For streamlined evaluation, use the included bash wrapper:

### Automated Pipeline (Colab/Linux)
You can run the entire evaluation process using the `run_eval.sh` script, which handles dataset splitting and batch size management dynamically.

**Step 1: Evaluate Original Model (Ground Truth)**
Generate baseline metrics without compression:
```bash
./run_eval.sh -t original -m gpt2-xl-code -b 16
```

**Step 2: Evaluate PCRL Agent**
Compare the compressed prompt generation against the baseline:
```bash
./run_eval.sh -t pcrl -c gpt2-xl-code -p PromptCompressor_code_RL/roberta_code_v1_bleu_512_part2 -b 16
```

### Manual Pipeline
If you prefer running Python scripts directly, follow this dependency sequence:

1. **Original Model:**
```bash
   python scripts/evaluate_original.py --gen_model=gpt2-xl-code --bs=16 --results_dir=results
   ```
2. **Heuristic Model (Optional baseline):**
```bash
   python scripts/evaluate_heuristic.py --gen_model=gpt2-xl-code --bs=16 --results_dir=results
   ```
3. **PCRL Model (Requires Original Model results):**
```bash
   python scripts/evaluate_pcrl.py --pcrl_model=PromptCompressor_code_RL/roberta_code_v1_bleu_512_part2 --seed=42 --gen_model=gpt2-xl-code --bs=16 --results_dir=results
   ```

## License
The codebase is licensed Apache 2.0 (see LICENSE). The data is a mixture of Self-Instruct (Apache 2.0) and Stanford Alpaca (CC BY-NC 4.0). By training on a mixture of the data, it inherits both licenses.

## Acknowledgements
This project is a fork of the original [PromptCompressor](https://github.com/nenomigami/PromptCompressor) repository. I express my gratitude to the authors for their foundational architecture and for making their code publicly available. 

Additionally, this work references and utilizes:
- [CodeAlpaca](https://github.com/sahil280114/codealpaca) for the code-generation instruction dataset.
- [RL4LMs by AllenAI](https://github.com/allenai/RL4LMs)
- [gisting by Mu et al.](https://github.com/jayelm/gisting)

## Contact
For any questions or feedback regarding this specific extension and evaluation pipeline, please contact dima.strizhak@gmail.com.