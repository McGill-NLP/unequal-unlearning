import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml
import os
from evaluate_util import eval_rougel_fast, get_lm_harness_results, evaluate_tf_prompt
from utils import free_gpu_memory
from datasets import load_dataset
from utils import set_deterministic
import argparse
from evaluate_util import eval_perplexity
#disabling wandb for eval
os.environ["WANDB_DISABLED"] = "true"


'''
This is a seperate evaluation script intended for evaluation that happens parallel to training 
The evaluation script expects a checkpoint directory path and a checkpoint step. The directory is expected to contain a config.yaml file.
This script then creates a results.jsonl file in the checkpoint directory, and logs result for the step. You can run several of these in parallel with different checkpoint steps numbers .
'''


#load hydra config
def evaluate_checkpoint(checkpoint_dir,checkpoint_step):  
    # read cfg from path
    checkpoint_name = f'checkpoint_{checkpoint_step}'
    cfg = OmegaConf.load(f'{checkpoint_dir}/config.yaml')
    dataset_cfg = cfg.dataset 
    #if wandb.init has not been initialized yet, initialize it
    # if not wandb.run:
    #     wandb.init(project='Unlearning', entity='siva-reddy-mila-org', name=f'{cfg.model_family}_{dataset_cfg.forget_split}_{cfg.loss_type}_{cfg.forget_loss_lambda}')
    #     wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    set_deterministic(seed = cfg.seed)
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    cfg.model_path = model_cfg["hf_key"]
    
    checkpoint_results = {}
    checkpoint_results['checkpoint'] = checkpoint_name
    print(f'Evaluating {checkpoint_dir}/{checkpoint_name}')

    if checkpoint_name == 'checkpoint_0':
        #checkpoint 0 is the base model
        utility = get_lm_harness_results(cfg.model_path, cfg.utility_tasks)
        checkpoint_results.update(utility)
        free_gpu_memory()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, 
                                                    device_map='auto')
        #Trick to make things faster
        model = torch.compile(model)
        model.half()
        model.eval()

    else:
        #these are the saved checkpoints
        utility = get_lm_harness_results(os.path.join(checkpoint_dir, checkpoint_name),cfg.utility_tasks)
        checkpoint_results.update(utility)
        free_gpu_memory()
        model = AutoModelForCausalLM.from_pretrained(os.path.join(checkpoint_dir, checkpoint_name),device_map='auto', attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, checkpoint_name))
        
        #Trick to make things faster
        model = torch.compile(model)
        model.half()
        model.eval()

    with torch.no_grad():
        #calculaing rougel scores
        forget_utility = eval_rougel_fast(model, tokenizer, dataset_cfg)
        checkpoint_results.update(forget_utility)

        #calculating perplexity
        perplexity = eval_perplexity(model, tokenizer, dataset_cfg, dataset_cfg.ppl_list)
        checkpoint_results.update(perplexity)

        #calculating forget quality using tf prompt
        paraphrased_forget_dataset = load_dataset(dataset_cfg.data_path, f'{dataset_cfg.forget_split}_paraphrased')['train']
        forget_tf_utility = evaluate_tf_prompt(paraphrased_forget_dataset, model, tokenizer, cfg.model_family)
        checkpoint_results.update(forget_tf_utility)

    #save results to a json file 
    with open(f'{checkpoint_dir}/results.jsonl', 'a') as f:
        json.dump(checkpoint_results, f)
        f.write('\n')
    
    print('DONE')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_step', type=int, required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint_dir, args.checkpoint_step)
