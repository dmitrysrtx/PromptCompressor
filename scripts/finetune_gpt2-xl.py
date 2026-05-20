from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import os
import wandb
import torch
import numpy as np


# PyTorch2.6 patch
# Saves an original load function
_original_load = torch.load

# Wrapper for weights_only=False
def _trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

# Exhange the original function of wrapper
torch.load = _trusted_load

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator


def train(train_file_path,
          model_name,
          output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def concat_instruction_input(examples):
        if examples["input"]:
            source = f"Instruction: {examples['instruction']}\nInput: {examples['input']}\nOutput: \n{examples['output']}<|endoftext|>"
        else:
            # No input, instruction only.
            source = f"Instruction: {examples['instruction']}\nOutput: \n{examples['output']}<|endoftext|>"
        inputs = tokenizer(source, max_length=512, truncation=True)
        return inputs

    train_dataset = load_dataset(train_file_path, split="train")
    train_dataset = train_dataset.map(concat_instruction_input, batched=False,remove_columns=["instruction","input","output", 'split'], num_proc=8)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
        
    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            # max_steps=50, # For env check only
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_bnb_8bit",
            save_strategy="steps", # Saves per step
            save_steps=500,        # backup for each 500 steps
            save_total_limit=1,    # Keeps only the last backup(disk space saver)
            report_to="wandb",     # Connecting to W&B
            run_name="gpt2-xl-alpaca-full", # Graph name
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
    )
        
    # Checks if any checkpoint is exist
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    # Starts training by providing the found checkpoint (if exists)
    if last_checkpoint is not None:
        print(f"🔄 Found checkpoint: {last_checkpoint}. Continue training!")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("🚀 Start trainig from scratch")
        trainer.train()

    trainer.save_model()

if __name__=="__main__":
    train_file_path = "data/alpaca_plus.py"
    model_name = "gpt2-xl"
    output_dir = 'gpt2-xl-finetuned'

    os.makedirs(output_dir, exist_ok=True)
    wandb_id_file = os.path.join(output_dir, "wandb_run_id.txt")
    
    last_checkpoint = get_last_checkpoint(output_dir)
    
    if last_checkpoint is not None and os.path.exists(wandb_id_file):
        # Read an old ID and say W&B to continue the session
        with open(wandb_id_file, "r") as f:
            run_id = f.read().strip()
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = run_id
        print(f"🔗 Stiching W&B graphs! Continue session: {run_id}")
    else:
        # If file doesn't exist, create new ID and save it
        run_id = wandb.util.generate_id()
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = run_id
        with open(wandb_id_file, "w") as f:
            f.write(run_id)
        print(f"🚀 New W&B session: {run_id}")

    train(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
    )