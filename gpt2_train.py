import torch
import wandb
import os
from transformers import Trainer
from transformers import GPT2LMHeadModel, AutoTokenizer, TrainingArguments, AutoConfig
import os
import hydra
from torch.distributed import is_initialized, barrier
from data_module import FakeBiographiesDataset
    
@hydra.main(config_path="config", config_name="gpt2_train", version_base=None)
def main(cfg):
    '''
    Train the GPT2 model on the fake biographies and qa data. Automatically detects if the model has been trained before and resumes from the last checkpoint.
    '''


    #check if the model has been trained before
    resume_from_checkpoint = bool(os.path.isdir(f'bios_models/{cfg.RUN_NAME}'))

    # Synchronize all processes to ensure they see the updated directory state
    if is_initialized():
        barrier()  # Wait for rank 0 to finish creating the directory

    print(f'\n\nResuming from checkpoint: {resume_from_checkpoint}\n\n')
    wandb.init(project="Unlearning-GPT2",name=cfg.RUN_NAME, group="Training")

    train_data = FakeBiographiesDataset(tokenizer_identifier="gpt2", data_path=cfg.DATA_PATH, SEED=cfg.RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    #Make the config for a fresh model
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=train_data.max_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation="sdpa",
        #feed everything from the config
        **cfg.gpt2
    )
    #create the model
    model = GPT2LMHeadModel(config)

    #move to cuda
    model = model.to("cuda")
    model.gradient_checkpointing_enable()

    #print the model size
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    #total number of steps, accounting for parallel training
    total_steps = cfg.EPOCHS * len(train_data) // (cfg.BATCH_SIZE * torch.cuda.device_count())

    training_args = TrainingArguments(
        output_dir=f"bios_models/{cfg.RUN_NAME}",
        per_device_train_batch_size=cfg.BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=cfg.EPOCHS,
        #warmup for 10% of the total steps
        warmup_steps=int(total_steps * 0.1),
    #   max_grad_norm=1.0,
        optim="adamw_torch_fused",
        learning_rate=cfg.LEARNING_RATE,
        logging_steps=1/(10*cfg.EPOCHS),
        logging_dir=f"bios_models/{cfg.RUN_NAME}/",
        #save after each epoch, provding decimal to scale with total number of steps
        save_steps=0.1,
        save_total_limit=-1,
        report_to="wandb",
        fp16=True,
        seed=cfg.RANDOM_SEED,
        dataloader_drop_last=True,
        deepspeed="config/ds_config.json",
        dataloader_num_workers = 12,
        dataloader_pin_memory=True,
        torch_compile=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    main()
