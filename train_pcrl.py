from argparse import ArgumentParser
from pcrl.utils.logging_utils import Tracker
from pcrl.utils.training_utils import OnPolicyTrainer
import os
import sys
sys.path.append('.')
import random
import yaml
import numpy as np
import torch

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool, 
    seed: int,
):

    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    set_seed_everywhere(seed)

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )

    warm_start_path = config.get("alg", {}).get("args", {}).pop("warm_start_path", "")
    
    trainer = OnPolicyTrainer(
        gen_config=config["gen_model"],
        datapool_config=config["datapool"],
        reward_config=config["reward_fn"],
        env_config=config["env"],
        alg_config=config["alg"],
        train_eval_config=config["train_evaluation"],
        tracker=tracker,
    )

    # ==========================================
    # 🔥 WARM START OR FROM SCRATCH LOGIC
    # ==========================================
    # Get the path from config. Default to empty string if not found.
    warm_start_path = config.get("alg", {}).get("args", {}).get("warm_start_path", "")
    
    # Check if a valid path was provided in the YAML
    if warm_start_path and isinstance(warm_start_path, str) and warm_start_path.strip() != "":
        # Check if the file/folder actually exists on the disk
        if os.path.exists(warm_start_path) or os.path.exists(warm_start_path + ".zip"):
            print(f"\n🚀 INITIALIZING WARM START...")
            print(f"📥 Loading agent weights from: {warm_start_path}")
            
            # Safely extract the algorithm object (SB3 MaskablePG)
            alg = getattr(trainer, 'alg', getattr(trainer, '_alg', getattr(trainer, 'model', None)))
            
            if alg is not None:
                try:
                    alg.set_parameters(warm_start_path)
                    print("✅ Weights loaded successfully! Agent will continue training.\n")
                except Exception as e:
                    print(f"❌ Error loading SB3 weights: {e}")
                    print("⚠️  Proceeding with training from scratch as a fallback.\n")
            else:
                print("❌ Error: Could not find the algorithm object inside OnPolicyTrainer.")
        else:
             print(f"\n❌ Error: Path '{warm_start_path}' does not exist.")
             print("⚠️  Proceeding with training FROM SCRATCH.\n")
    else:
        print("\n🌱 INITIALIZING TRAINING FROM SCRATCH (No warm start path provided).\n")
    # ==========================================
    
    trainer.train_and_eval()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--project_name", type=str, help="WANDB project name", default="pcrl_exps"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="WANDB experiment name",
        default="llama2_2023",
    )
    parser.add_argument(
        "--entity_name", type=str, help="WANDB entity name",
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--seed", type=int, help="random seed to use", default=2023
    )
    args = parser.parse_args()

    main(
        args.config_path,
        args.project_name,
        args.experiment_name,
        args.base_path_to_store_results,
        args.entity_name,
        args.log_to_wandb,
        args.seed,
    )
