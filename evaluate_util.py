from tqdm.auto import tqdm
import torch
import os
from torch.nn import CrossEntropyLoss
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml
import numpy as np 
from datasets import load_dataset
import wandb
import os
import subprocess
import wandb
import random
import uuid
from utils import make_biography
from transformers import pipeline
import logging
logging.getLogger('absl').setLevel(logging.ERROR)


def compute_rouge_l(reference, prediction):
    ''' Compute ROUGE-L score between reference and prediction '''
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #for the synthetic setup, there might be a training period at the end of the answer. We remove it.
    prediction = prediction.replace('.', '').replace(',','').strip().lower()
    reference = reference.replace('.', '').replace(',','').strip().lower()
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].recall
                    

def eval_rougel_fast(model, tokenizer, cfg):
    ''' Main evaluation function. Runs all the QA evaluation tasks where we generate answers and compute rougeL recall
    Evaluates on the split list from the dataset config file (and eval_task for key-names)
    '''
    model.eval()
    tokenizer.padding_side = "left"

    # ✅ Move model to GPU and use mixed precision
    device = model.device
    eval_results = {}

    with torch.no_grad():
        for split, eval_task in zip(cfg['split_list'], cfg['eval_task']):
            print(f'🚀 Evaluating: {eval_task} on {split}')
            if split == 'flan_few_shot':
                max_new_tokens = 15
            else:
                max_new_tokens = 50
            
            data = load_dataset(cfg['data_path'], split)["train"]

            # ✅ Add chat template from the model config
            model_configs = get_model_identifiers_from_yaml(cfg['model_family'])
            question_start, question_end = model_configs['question_start_tag'], model_configs['question_end_tag']
            #making completion prompts with the chat templates
            completion_prompts = [question_start + q.strip() + question_end for q in data['question']]
            references = data['answer']

            generations = []
            batch_size = cfg['eval_batch_size']

            for i in tqdm(range(0, len(completion_prompts), batch_size)):
                batch = completion_prompts[i:i + batch_size]

                # ✅ Tokenize with dynamic padding and move to GPU
                tokenized_batch = tokenizer(batch, padding="longest", truncation=True, return_tensors="pt").to(device)

                # ✅ Generate responses 
                outputs = model.generate(**tokenized_batch, max_new_tokens=max_new_tokens, do_sample=False)

                # ✅ Decode
                batch_generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generations.extend(batch_generations)

            # ✅ Compute ROUGE scores 
            eval_results[eval_task] = np.mean(np.vectorize(compute_rouge_l)(references, generations))

    print(eval_results)
    if wandb.run is not None:
        wandb.log(eval_results)
    return eval_results


######################################################## OLMo Evaluation ########################################################

#All QA style evaluations (Utility, forget quality, FLAN, world facts etc.) use the eval_rougel_fast function. The following functions are for other evaluation tasks: True/False Prompting, Perplexity and Probabilistic Evaluation


def get_lm_harness_results(model_path, tasks = ['tinyTruthfulQA', 'tinyWinogrande', 'tinyHellaswag', 'tinyMMLU', 'tinyGSM8k']):
    '''
    Get results from lm-eval-harness. This is to evaluate tinyMMLU and TinyHellaswag
    '''
    output = subprocess.check_output(f"python -m lm_eval --model hf --model_args pretrained={model_path} --tasks={','.join(tasks)} --batch_size=2 --device=cuda:0", shell=True).decode("utf-8")
    results = {}
    for line in output.split('\n')[3:-2]:
        task = line.split('|')[1].strip()
        metric = line.split('|')[5].strip()
        value = float(line.split('|')[7].strip())
        results[f'eval_{task}'] = value
    #assert len(results) == len(tasks), f"Expected {len(tasks)} tasks, got {len(results)}"
    print('Utility: ', results)
    if wandb.run is not None:
        wandb.log(results)
    return results

def generate_tf_prompt(dataset, model_family):
    ''' Generate a few-shot True or False prompt for each entry in the dataset. 
    The answer is always true, but this is okay because the few shot prompts have false answers as well'''
    prompts = []
    answers = []
    few_shot_prompt = (
        "The following are Questions and Answers. State if the answer to each question is true or false.\n\n"
            "Question: How many planets are in the solar system?\nAnswer: Eight\nTrue or False: True\n\n"
            "Question: What is the tallest mountain in the world?\nAnswer: Mount Everest\nTrue or False: True\n\n"
            "Question: Where was pizza invented?\nAnswer: France\nTrue or False: False\n\n"
            "Question: Who painted the Mona Lisa?\nAnswer: Leonardo da Vinci\nTrue or False: True\n\n"
            "Question: What color is the gemstone ruby?\nAnswer: Blue\nTrue or False: False\n\n"
            "Question: Are sharks mammals?\nAnswer: No\nTrue or False: False\n\n"
            "Question: Can humans breathe underwater without equipment?\nAnswer: Yes\nTrue or False: False\n\n"
            "Question: Who wrote the novel *1984*?\nAnswer: George Orwell\nTrue or False: True\n\n"
            "Question: What is the chemical symbol 'Au' for?\nAnswer: Aluminum\nTrue or False: False\n\n"
            "Question: What is the primary ingredient in traditional hummus?\nAnswer: Chickpeas\nTrue or False: True\n\n"
)
    # Loop through each entry in the dataset
    #get model identifiers
    model_cfg = get_model_identifiers_from_yaml(model_family)
    for entry in dataset:
        question = entry['question']
        answer = entry['answer']
        prompt = f"Question: {question}\nAnswer: {answer.capitalize()}\nTrue or False:"
        prompts.append(few_shot_prompt + prompt)   
        answers.append('True')
    return prompts, answers


def evaluate_tf_prompt(dataset,model,tokenizer, model_family):
    ''' Evaluate the True or False prompt on the dataset. Pass a dataset with "question" and "answer" columns.
    1. Script Constructs a few-shot True or False-MCQ prompt for the model to process
    2. Compares probability diff between option True,False and checks if the model's answer matches the correct answer 
    3. Basically changes eval from a generative one to a contrastive one, you know for funsies'''
    # Generate the prompts
    tf_prompts,answers = generate_tf_prompt(dataset, model_family)
    tokenized_inputs = [torch.as_tensor([i]) for i in tokenizer.batch_encode_plus(tf_prompts).input_ids]
    # Get the token IDs for each of the Answer tokens. 
    # Hello is just a placeholder to get the correct token for the first option
    answer_token_ids = tokenizer.encode("Hello True False",add_special_tokens=False)[-2:]

    device = model.device
    model.eval()  # Ensure the model is in eval mode

    model_answers=[]
    for tokenized_input in tqdm(tokenized_inputs):    
        tokenized_input = tokenized_input.to(model.device)
        # Forward pass 
        with torch.no_grad():
            outputs = model(tokenized_input)
            logits = outputs[0]
        # Extract logits for the specified tokens
        last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        logits_comparison = last_token_logits[0, answer_token_ids]  # Shape: (len(tokens_to_compare),)        
        model_answers.append(torch.argmax(logits_comparison).item())
    
    #we only need to check if the probability of True is greater than Falseq
    tf_acc = np.mean([i==0 for i,j in zip(model_answers,answers)])
    print(f"Accuracy on TF: {tf_acc} \n")
    if wandb.run is not None:
        wandb.log({"forget_paraphrased_TF": tf_acc})
    return {"eval_forget_paraphrased_TF": tf_acc}


def eval_perplexity(model, tokenizer, cfg, split_list):
    '''Evaluate perplexity on the dataset. For OLMo, the input is the Tulu subset'''
    model.eval()
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    eval_results={}

    #convert split_list to list if it is a string
    if isinstance(split_list, str):
        split_list = [split_list]
        
    for i, split in enumerate(split_list):
        print(f'Working on Perplexity for {split}')
        dataset = load_dataset(cfg.data_path,split)['train']

        #formatting instances ith the chat template before we compute perplexity. The instruction tuning samples must have look like this
        dataset = dataset.map(lambda x: {'formatted_text': f'{model_cfg["question_start_tag"]}{x["question"]}{model_cfg["question_end_tag"]}{model_cfg["answer_tag"]}{x["answer"]}'}, num_proc=8)

        #measure perplexity on these instances
        results = compute_perplexity(predictions=dataset['formatted_text'], 
                             model=model, 
                             tokenizer=tokenizer,
                             add_start_token=False, 
                             batch_size=cfg.eval_batch_size)
        eval_results[f'eval_perplexity_{split}'] = results['mean_perplexity']
    print(eval_results)
    if wandb.run is not None:
        wandb.log(eval_results)
    return eval_results


def compute_perplexity(
    predictions, batch_size: int = 16, add_start_token: bool = True, device=None, model=None, tokenizer=None, max_length=None
):
    ''' This is a slightly modified version of the huggingface perplexity computation. See https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py '''
    model.eval()
    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def submit_eval_job(checkpoint_dir, checkpoint_step):
    '''
    Submit evaluation job to the cluster
    '''
    job_script = f"""#!/bin/bash
    #SBATCH --partition=long
    #SBATCH --cpus-per-task=6
    #SBATCH --mem=32GB
    #SBATCH --time=2:00:00
    #SBATCH --gres=gpu:1
    #SBATCH --array=1
    #SBATCH --output=/home/mila/k/krishnaa/scratch/logs/output/output_%A_%a.log  # %A for array job ID, %a for array task ID
    #SBATCH --error=/home/mila/k/krishnaa/scratch/logs/error/error_%A_%a.log    # Adjust the path as needed

    # Load any required modules.
    module load miniconda/3 cuda/12.4.0
    conda activate tofu2

    # Install your project dependencies.
    cd "/home/mila/k/krishnaa/scratch/TOFU"

    echo "Evaluating checkpoint_{checkpoint_step} at {checkpoint_dir}"

    python unlearn_eval.py --checkpoint_dir {checkpoint_dir} --checkpoint_step {checkpoint_step}
        """
    random_file_name = f"unlearn_eval_{random.randint(1, 1000000)}.slurm"
    with open(f"{random_file_name}", "w") as f:
        f.write(job_script)
    subprocess.run(["sbatch", random_file_name])        
    os.remove(random_file_name)



######################################################## GPT-2 Evaluation ########################################################

#For QA evaluation, we use the eval_rougel_fast function. The BIO evaluation script can be found below



def make_completion_prompt(row, attribute):
    '''Make a completion prompt for the BIO-Accuracy evaluation by splitting at the answer position'''
    answer = row[attribute]
    completion_prompt = row['BIOGRAPHY'].split(answer)[0].strip()
    return {'completion_prompt':completion_prompt}


def eval_completion(model, tokenizer, completion_prompts, answers, batch_size=512):
    ''' 
    This is the evaluation of the BIO-Accuracy of the model
      '''
    #for each row, create a new row with the completion prompt and the answer\
    #add eos token to the completion prompt
    completion_prompts = [tokenizer.eos_token + completion_prompt.strip() for completion_prompt in completion_prompts]
    #generate the completion
    generations = []
    references = []
    
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # Process in batches
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer, 
                    max_new_tokens=10, 
                    do_sample=False,
                    batch_size=batch_size) # Added batch size

    # Process batches of questions
    progress_bar = tqdm(range(0, len(completion_prompts), batch_size))
    for i in progress_bar:
        batch = completion_prompts[i:i+batch_size]
        # Generate responses for batch
        outputs = pipe(batch)
        
        # Extract generated text from each output
        batch_generations = [out[0]['generated_text'] for out in outputs]

        # Store results
        generations.extend(batch_generations)
        references.extend(answers[i:i+batch_size])
        progress_bar.update(batch_size)
    print(list(zip(completion_prompts,generations,references))[0])
    return np.mean([compute_rouge_l(ref,gen) for gen,ref in zip(generations,references)])


def run_biography_completion_evaluation(dataset_cfg, model, tokenizer, construct_new_biographies=True):
    #get the attribute from the forget split    
    #EXPECTED_FORMAT is 'forget_high_count_qa_birthday'. We split it into forget_attribute and forget_split
    forget_attribute = dataset_cfg['forget_split'].split('_')[-1].upper()
    forget_split = '_'.join(dataset_cfg['forget_split'].split('_')[:-2])

    #make sure tokenizer is left padded
    tokenizer.padding_side = 'left'

    #Turn this ON to evaluate over all attributes for BIO evaluation. This gives you a 'neighborhood evaluation' for the target attribute. T
    # hat is, if you are unlearning the employer attribute for the target split (say low-count), this gives a BIO accuracy for the other attributes of that split, 
    #to make sure that the other attributes are not being affected by the unlearning.
    #attributes_to_evaluate = ['UNIVERSITY', 'MAJOR', 'EMPLOYER', 'BIRTHDAY', 'LOCATION']
    attributes_to_evaluate = [forget_attribute]
    #to store compiled results
    log_results = {}

    for split in [forget_split, 'utility', 'retain']:  
        #to store intermediate results across attributes
        results = {}     
        #we make forget_high_count into forget and keep the rest of the split name the same  
        split_name = split.split("_")[0]

        dataset = load_dataset(dataset_cfg["data_path"], f'{split}_biography')['train']
        #high count has upsampled duplicate biographies. So we select only subset
        if split == 'forget_high_count':
            dataset = dataset.select(range(len(set(dataset['NAME']))))
        #We recreate biographies by random sampling, to make sure that the evaluation is robust
        if construct_new_biographies:
            dataset = dataset.map(make_biography, num_proc=2, new_fingerprint=str(uuid.uuid4()), remove_columns=['BIOGRAPHY'])

        for attribute in attributes_to_evaluate:
            print(f'Running Biography Evaluation for {split} and {attribute}')            
            dataset = dataset.map(make_completion_prompt, fn_kwargs={'attribute':attribute}, num_proc=8)
            #filter out the rows where completion_prompt is ''
            dataset = dataset.filter(lambda x: x['completion_prompt'] != '')
            #for each row, create a new row with the completion prompt and the answer
            results[f'{split_name}_{attribute}_bio_evaluation'] = eval_completion(model, tokenizer, dataset['completion_prompt'], dataset[attribute], batch_size=1024)
        
        #log the forget_attribute seperately
        log_results[f'{split_name}_bio_evaluation'] = results[f'{split_name}_{forget_attribute}_bio_evaluation']
        #lets pop the forget_attribute from the results
        results.pop(f'{split_name}_{forget_attribute}_bio_evaluation')
        #now, we should average the rest of the results for neighborhood evaluation
        log_results[f'{split_name}_bio_neighborhood_evaluation'] = np.mean(list(results.values()))

    if wandb.run is not None:
        wandb.log(log_results)
    return log_results






