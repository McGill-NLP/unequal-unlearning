# Not All Data Are Unlearned Equally

Code and documentation for the paper [Not All Data Are Unlearned Equally](https://arxiv.org/pdf/2504.05058)

## Installation

```
conda create -n unlearn python=3.10
conda activate unlearn
pip install -r requirements.txt
```
Make sure you set up the wandb configurations in the configuration files for training and unlearning. 


## lm-eval-harness
To run the probabilistic experiments for OLMo (MMLU, Hellaswag), make sure that [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) is installed in this directory. The evaluation script calls it as 
```
python -m lm_eval --options
```
You can also turn these evaluations by editing ```config/unlearn.yaml```


## Datasets

1. We release the code to sample biography datasets for the GPT-2 experiments. The dataset can be created by running the `biographies.ipynb` notebook, which randomly samples a new dataset and uploads it to huggingface. We refrain from publically releasing the datasets we use following [[privacy concerns]](https://physics.allen-zhu.com/faq#h.302roqlyumwu) raised by the original authors. 
2. The OLMo datasets, seperated by count buckets are released [here](https://huggingface.co/collections/McGill-NLP/unequal-unlearning-67ff85082467b37f6f2bb20e).


## Finetuning GPT2

the gpt2_bios.py and the config/gpt_train.yaml file contains the code for finetuning GPT2 on the fake biographies and questions. For GPU-parallel training, use

```
torchrun --nproc_per_node=4 gpt2_bios.py gpt2=124M LEARNING_RATE=1e-3 SCALE_FACTOR=10 EPOCHS=100 BATCH_SIZE=32 RANDOM_SEED=1234
```



## Unlearn models
Make sure that the model identifier is correctly provided in the `config/model_config.yaml` file. To unlearn a model on a forget set, use the following command:
```
# python unlearn_train_eval.py \
#   --mask_retain_question False \
#   --loss_type SIMNPO_GD \
#   --dataset books \
#   --dataset.forget_split forget_high_count \
#   --dataset.retain_split forget_medium_count \
#   --model_family olmo-7b-sft \
#   --learning_rate 5e-6 \
#   --batch_size 8 \
#   --forget_loss_lambda 1.0 \
#   --retain_loss_lambda 5.0 \
#   --simnpo_gamma 0 \
#   --npo_beta 0.1 \
#   --unlearn_epochs 5 \
#   --seed 1`
```

You can choose between SIMNPO, GA and IDK for the loss function. Results are automatically uploaded to wandb.

## Evaluate models

The `unlearn_train.py` script automatically runs evaluation after each epoch. Incase you wish to make things faster by evaluating things in parallel, `unlearn_eval.py` accepts a directory and evaluates the model at this directory. The path should contain a yaml file with all necessary information to load the model, the dataset and the evaluation tasks.
Example command:
```
python unlearn_eval.py --checkpoint_dir path/to/checkpoint --checkpoint_step 100
```


