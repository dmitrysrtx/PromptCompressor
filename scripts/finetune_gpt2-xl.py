from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
from transformers.trainer_utils import get_last_checkpoint


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
            max_steps=50,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_bnb_8bit",
            save_strategy="steps", # Saves per step
            save_steps=500,        # backup for each 500 steps
            save_total_limit=3,    # Keeps last 3 backups(disk space saver)
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
    train(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
    )