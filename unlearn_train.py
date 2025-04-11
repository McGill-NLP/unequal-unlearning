import os
import random
import wandb
import hydra
import shutil
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml
from torch.amp import autocast
from losses import get_loss
from data_module import ForgetDataset, custom_data_collator_forget
from evaluate_util import eval_perplexity, eval_rougel_fast, get_lm_harness_results, run_biography_completion_evaluation, evaluate_tf_prompt
from utils import get_linear_warmup_scheduler
from tqdm.auto import tqdm
from omegaconf import DictConfig


'''
Main unlearning script. Configurations taken from config/unlearn.yaml and config/dataset/*.yaml
example config:
model_family: olmo-7b-sft
dataset.retain_split: 'forget_medium_count'
dataset.forget_split: 'forget_high_count'
learning_rate: 2e-6
unlearn_epochs: 10
batch_size: 16
loss_type: GA_GD
'''

def set_deterministic(seed=42):
    # Ensure deterministic operations where possible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#load hydra config
@hydra.main(version_base=None, config_path='config', config_name='unlearn')
def main(cfg: DictConfig):
    set_deterministic(seed = cfg.seed)
    dataset_cfg = cfg.dataset  

    wandb.init(project=dataset_cfg.wandb.project, 
               entity=dataset_cfg.wandb.entity, 
               name=f"{dataset_cfg.dataset_name}_{cfg.model_family}_{dataset_cfg.forget_split}_{cfg.loss_type}_{cfg.forget_loss_lambda}_{cfg.retain_loss_lambda}",
               group=dataset_cfg.wandb.group
            #    mode = "disabled"  
)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.log(OmegaConf.to_container(cfg, resolve=True))
    
    #Get model config 
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    Optimizer = torch.optim.AdamW

    #If a model path is given, use it, otherwise load the model from the config default path
    if cfg.model_path is None:
        cfg.model_path = model_cfg["hf_key"]

    #Model and tokenizer load
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, 
                                                device_map='cuda',
                                                attn_implementation="sdpa",
                                                torch_dtype=torch.bfloat16)
    
    print(f'Unlearning on {dataset_cfg.dataset_name}, split {dataset_cfg.forget_split} regularized with {cfg.loss_type} loss on {dataset_cfg.retain_split} split')

    #Make directory for saving checkpoints and the config files
    save_path = f'unlearned_models/{cfg.model_family}/ngram_counts/{dataset_cfg.dataset_name}/{cfg.loss_type}_{cfg.retain_loss_lambda}_{cfg.seed}/{dataset_cfg.forget_split}'
    if os.path.exists(f'{save_path}'):
        shutil.rmtree(f'{save_path}')
    os.makedirs(f'{save_path}/checkpoints')
    #save hydra config, incase you want to run evaluation in parallel to training (which is what we did to save time; see unlearn_eval.py)
    OmegaConf.save(cfg, f'{save_path}/checkpoints/config.yaml')

    #Dataset and dataloader creation
    torch_format_dataset = ForgetDataset(dataset_cfg.data_path, 
                                         tokenizer=tokenizer, 
                                         model_family = cfg.model_family, 
                                         max_length=dataset_cfg.generation.max_length, 
                                         forget_split=dataset_cfg.forget_split, 
                                         retain_split=dataset_cfg.retain_split, 
                                         loss_type=cfg.loss_type,
                                         mask_retain_question=cfg.mask_retain_question)
    dataloader = DataLoader(torch_format_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_data_collator_forget)  # No shuffling for determinism
    #we need the paraphrased dataset for tf evaluation
    if dataset_cfg.tf_evaluation:
        paraphrased_forget_dataset = load_dataset(dataset_cfg.data_path, f'{dataset_cfg.forget_split}_paraphrased')['train']

    #################### Unlearning ####################
    # Calculate the number of warmup steps and total steps
    total_steps = cfg.unlearn_epochs * len(dataloader)  # Total steps = epochs * steps per epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps


    # Loss function and optimizer
    optimizer = Optimizer(model.parameters(), lr= cfg.learning_rate)  # Using SGD optimizer 
    # Initialize the lr scheduler
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
 
    ##### Evaluate before Unlearning ######
    #run biography evaluation if applicable
    if dataset_cfg.biography_evaluation: run_biography_completion_evaluation(dataset_cfg, model, tokenizer)    
    #run perplexity evaluation if applicable
    if dataset_cfg.ppl_list: eval_perplexity(model, tokenizer, dataset_cfg, dataset_cfg.ppl_list)
    #run probabilistic evaluation with lm-harness if applicable
    if dataset_cfg.utility_tasks: get_lm_harness_results(cfg.model_path, dataset_cfg.utility_tasks)
    #run tf evaluation if applicable
    if dataset_cfg.tf_evaluation: evaluate_tf_prompt(paraphrased_forget_dataset, model, tokenizer, cfg.model_family)
    #qa evaluation
    eval_rougel_fast(model, tokenizer, dataset_cfg)


    ##### Loop through epochs #####
    progress_bar = tqdm(total=total_steps, desc="Unlearning Progress")
    step = 0
    for epoch in range(cfg.unlearn_epochs):

        for batch in dataloader:
            model.train()
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            with autocast('cuda',dtype=torch.bfloat16):  # Enable bfloat16 precision
                #get the loss terms
                forget_answer_loss, forget_question_loss, regularization_loss = get_loss(model, batch, cfg.loss_type, cfg.npo_beta,cfg.simnpo_gamma)
                #compute total loss
                loss = cfg.forget_loss_lambda * forget_answer_loss + cfg.forget_loss_lambda_question * forget_question_loss + cfg.retain_loss_lambda * regularization_loss
            #backpropagate
            loss.backward()

            #track loss components, learning rate and training metrics
            wandb.log({'loss': loss.item(), 'forget_answer_loss': forget_answer_loss.item(), 'forget_question_loss': forget_question_loss.item(), 'regularization_loss': regularization_loss.item()})
            wandb.log({'learning_rate': optimizer.param_groups[0]["lr"], 'checkpoint': step})
            
            # Step the scheduler and optimizer
            optimizer.step()
            scheduler.step()

            #update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            step += 1

        ### Evaluate and Save after each epoch ###
        #evaluate
        if dataset_cfg.biography_evaluation: run_biography_completion_evaluation(dataset_cfg, model, tokenizer)    
        if dataset_cfg.ppl_list: eval_perplexity(model, tokenizer, dataset_cfg, dataset_cfg.ppl_list)
        if dataset_cfg.utility_tasks: get_lm_harness_results(cfg.model_path, dataset_cfg.utility_tasks)
        if dataset_cfg.tf_evaluation: evaluate_tf_prompt(paraphrased_forget_dataset, model, tokenizer, cfg.model_family)
        eval_rougel_fast(model, tokenizer, dataset_cfg)

        #save model, tokenizer
        # model.save_pretrained(f'{save_path}/checkpoints/checkpoint_{step}')
        # tokenizer.save_pretrained(f'{save_path}/checkpoints/checkpoint_{step}')

if __name__ == '__main__':
    main()
