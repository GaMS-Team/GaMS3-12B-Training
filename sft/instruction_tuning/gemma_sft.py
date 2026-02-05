from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

from argparse import ArgumentParser
import os


def run_training(experiment_dir, model_output_path, model_input_path, run_name, resume_from_checkpoint=None):
    # Load the datasets from JSONL files
    train_dataset = load_dataset("json", data_files=f"/data/training.jsonl")["train"]
    val_dataset = load_dataset("json", data_files=f"/data/validation.jsonl")["train"]

    # Load tokenizer (you'll need this for the data collator)
    tokenizer = AutoTokenizer.from_pretrained(model_input_path)

    # Adjust micro batch size based on your hardware
    micro_batch_size = 8
    batch_size = 64

    world_size = int(os.environ["WORLD_SIZE"])
    gradient_accumulation_steps = batch_size // (world_size * micro_batch_size)

    print("Micro batch size:", micro_batch_size)
    print("Effective batch size:", batch_size)
    print("Gradient accumulation steps:", gradient_accumulation_steps)
    print("World size:", world_size)

    num_epochs = 3
    steps_per_epoch = len(train_dataset) // batch_size
    eval_steps = int(1 / 3 * steps_per_epoch)  # Evaluate 3 times per epoch
    save_steps = int(1 / 3 * steps_per_epoch)  # Save 3 times per epoch
    warmup_steps = 1000

    print("--------------------------------")
    print("Training parameters:")
    print("--------------------------------")
    print(f"Run name: '{run_name}'")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Evaluate each {eval_steps} steps ({eval_steps / steps_per_epoch:.2f} epochs)")
    print(f"Save each {save_steps} steps ({save_steps / steps_per_epoch:.2f} epochs)")
    print(f"Warmup steps: {warmup_steps} steps ({warmup_steps / steps_per_epoch:.2f} epochs)")
    print("--------------------------------")

    print("Initializing training args")
    training_args = SFTConfig(
        num_train_epochs=num_epochs,
        data_seed=5,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        logging_strategy="steps",
        logging_dir=f"{experiment_dir}/logs",
        logging_first_step=True,
        output_dir=experiment_dir,
        prediction_loss_only=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        logging_steps=10,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-6,
        warmup_steps=warmup_steps,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-6},
        bf16=True,
        dataloader_num_workers=8,
        push_to_hub=False,
        gradient_checkpointing=True,
        deepspeed="/script/deepspeed_config.json",
        max_length=131072,
        completion_only_loss=True
    )
    print("Training args initialized")

    print("Initializing model")
    model = AutoModelForCausalLM.from_pretrained(
        model_input_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    print("Model initialized")

    # Define Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Resume logic
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()

    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the dir where logs and checkopints will be stored."
    )
    parser.add_argument(
        "--model_input_path",
        type=str,
        required=True,
        help="Name of the input model."
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        required=True,
        help="Path to the dir where trained model will be saved."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.experiment_dir, args.model_output_path, args.model_input_path, args.run_name,
                 args.resume_from_checkpoint)
