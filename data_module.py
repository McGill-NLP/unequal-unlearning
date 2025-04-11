import torch
from torch.utils.data import Dataset
from utils import get_model_identifiers_from_yaml, add_dataset_index
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def custom_data_collator_forget(samples):
    rets = []
    '''
    Collate function for the training data. We load one forget instance and one retain instance during each step
    '''
    # Extracting samples for each data type
    data_types = ["forget", "retain"]
    samples_dict = {data_type: [sample[i] for sample in samples] for i, data_type in enumerate(data_types)}

    for data_type in data_types:
        data = samples_dict[data_type]

        input_ids = [s[0] for s in data]
        question_label = [s[1] for s in data]
        answer_label = [s[2] for s in data]
        attention_mask = [s[3] for s in data]

        rets.append((torch.stack(input_ids), torch.stack(question_label), torch.stack(answer_label), torch.stack(attention_mask)))

    return rets

def custom_data_collator(samples):
    '''Collator function for the QA evlaution samples'''
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs,mask_question=True, data_type='retain'):
    '''THis takes care of all the masking and so on for loss computation    
    The args are:
    tokenizer: the tokenizer to use
    max_length: the length of the instances. Batches are made by padding to this length
    question: the question 
    answer: the answer
    model_configs: the model configurations
    mask_question: whether to mask the question for the retain and forget instances. This adds the additional regularization terms in our loss function 
    data_type: whether the data is for forget or retain
    '''
    #add a question label and an answer label for the forget instance
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    #counting question tokens, we have to mask them later
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )

    #this is used to pad batches
    pad_length = max_length - len(encoded.input_ids)

    #padding with eos token and attention mask
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        #for the forget instance, we don't backprop the eos token. 
        if data_type == "forget":
            label = encoded['input_ids'] +  [-100] * (pad_length)
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    answer_label = label.copy()
    question_label = label.copy()
    #change label to -100 for question tokens if mask_question is true
    if mask_question:
        for i in range(num_question_tokens): answer_label[i] = -100
        for i in range(num_question_tokens, len(label)): question_label[i] = -100

    assert len(label) == len(pad_input_ids) == len(pad_attention_mask)
    return torch.tensor(pad_input_ids),torch.tensor(answer_label),torch.tensor(question_label),torch.tensor(pad_attention_mask)

    

class FakeBiographiesDataset(Dataset):
    def __init__(self, tokenizer_identifier: str, data_path: str, SEED: int, max_length: int = 512):
        """
        Dataset class that makes data for pretraining. 2 considerations from the paper:
        1. Several instances are joined together to form a single instance. The paper uses 512 token-length chunks.
        2. The Biography and QA data is trained together (Not in a single instance, but in a single batch)
        3. The token ratio is fixed at 1:3 (biography:qa)
        Args:
            tokenizer_identifier: The identifier of the tokenizer to use for encoding the text
            max_length: Maximum sequence length
        """
        # Load the dataset from Hugging Face
        
        self.biography_dataset = load_dataset(data_path, "fake_biographies_train")["train"]
        self.qa_dataset = load_dataset(data_path, "fake_biographies_qa_train")["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)


        print ('length of biography dataset', len(self.biography_dataset))
        print ('length of qa dataset', len(self.qa_dataset))
        print('total dataset length', len(self.biography_dataset) + len(self.qa_dataset))
        #tokenize the text
        self.biography_dataset = self.biography_dataset.map(self.tokenize_text,input_columns='BIOGRAPHY',num_proc=3,remove_columns=self.biography_dataset.column_names, load_from_cache_file=False)
        self.qa_dataset = self.qa_dataset.map(self.tokenize_text,input_columns='qa',num_proc=3,remove_columns=self.qa_dataset.column_names, load_from_cache_file=False)
        self.index_card = [f'BIOGRAPHY_{i}' for i in range(len(self.biography_dataset))] + [f'QA_{i}' for i in range(len(self.qa_dataset))]



        #the number of qa instances is 3x the number of biography instances
        required_qa_length = len(self.biography_dataset) * 3
        num_qa_instances_to_add = required_qa_length - len(self.qa_dataset)
        #add random instances from the qa dataset until we have the required number
        print(f'Adding {num_qa_instances_to_add} qa instances')
        for i in range(num_qa_instances_to_add):
            random_index = np.random.randint(0, len(self.qa_dataset))
            self.index_card.append(f'QA_{random_index}')

             
        #SHUFFLE CARD
        random.shuffle(self.index_card)
        self.max_length = max_length
        self.biography_dataset = self.biography_dataset['input_ids']
        self.qa_dataset = self.qa_dataset['input_ids']
        np.random.seed(SEED)
    
    def __len__(self):
        return len(self.index_card)

    def __getitem__(self, idx):
        #choose qa or biography
        dataset_name, index = self.index_card[idx].split('_')
        dataset = self.biography_dataset if 'BIOGRAPHY' == dataset_name else self.qa_dataset
        index = int(index)

        #get a random index
        random_indices = np.random.randint(0, len(dataset), size=20 if 'BIOGRAPHY' == dataset_name else 50)
        #now make instance
        input_ids = self.construct_instance(dataset[index], [dataset[i] for i in random_indices])

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids  # All tokens are valid
        }

    def tokenize_text(self, text):
        formatted_text = self.tokenizer.eos_token + text
        encoded_text = self.tokenizer.encode(formatted_text,padding=False)
        return {'input_ids':torch.tensor(encoded_text,dtype=torch.long)}

    def construct_instance(self,instance, context):
        instance = list(instance)
        while len(instance) < 512:
            random_index = np.random.randint(0, len(context))
            random_seq = context[random_index]
            instance.extend(random_seq)
        return torch.tensor(instance[:self.max_length],dtype=torch.long)


class ForgetDataset(Dataset):
    '''
    Datset instance that loads the forget and retain data during unlearning
    '''
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, forget_split = "forget05", retain_split = "real_authors", loss_type="GA_GD", mask_retain_question=True):
        super(ForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.forget_data = load_dataset(data_path, forget_split)["train"]
        self.retain_data = load_dataset(data_path, retain_split)["train"]

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        self.mask_retain_question = mask_retain_question
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ['forget', 'retain']:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else torch.randint(0, len(data), (1,)).item()
            question = data[idx]['question']

            if 'idk' in self.loss_type.lower() and data_type == 'forget':
                answer = "I do not know the answer"
            else:
                answer = data[idx]['answer']

            mask_question = self.mask_retain_question if data_type == "retain" else True
            converted_data = convert_raw_data_to_model_format(self.tokenizer, 
                                                              self.max_length, 
                                                              question, 
                                                              answer, 
                                                              self.model_configs,
                                                              mask_question,
                                                              data_type)
            rets.append(converted_data)
        return rets



class TextDatasetQA(Dataset):
    '''
    Dataset instance that loads the text data for QA style UTILITY evaluations
    '''
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)
