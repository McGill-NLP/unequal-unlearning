import torch
import torch.nn as nn
import torch.nn.functional as F

'''
These functions are used to compute the loss for the unlearning process.
The loss functions are defined in the config file.

We expect the loss to be described in the format forget-loss_retain-loss (eg: GA_GD)
The loss is then computed in the get_loss function. 
You can easily add new loss functions by adding a individual loss function and updating the get_loss function.
'''

########## Main Loss Function ##########

def get_loss(model, inputs, loss_type, npo_beta = 0.1,gamma = 0.0):
    '''
    This function computes the loss for the unlearning process.
    It takes in a model, inputs, loss_type, ref_model, npo_beta, gamma.
    It returns the forget_answer_loss, forget_question_loss, regularization_loss.
    '''
    # forget_loss
    if 'GA' in loss_type:
        forget_answer_loss, forget_question_loss = ga_loss(model, inputs)
    elif 'SIMNPO' in loss_type:
        forget_answer_loss, forget_question_loss = simnpo(model, inputs,npo_beta,gamma)
    elif 'IDK' in loss_type:
        forget_answer_loss, forget_question_loss = idk_loss(model, inputs)
    else:
        forget_answer_loss, forget_question_loss = 0, 0

    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    else:
        regularization_loss = 0
    return forget_answer_loss, forget_question_loss, regularization_loss


########## Individual Loss Functions ##########

# Forget Loss: GA
def ga_loss(model, inputs):
    device = model.device
    # The first element of the data tuple is the target data
    forget_inputs = inputs[0]
    input_ids, answer_label, question_label, attention_mask = forget_inputs
    # Compute the Cross entropy loss for the answer
    outputs = model(input_ids.to(device), labels=answer_label.to(device), attention_mask=attention_mask.to(device))
    #reversing the sign for gradient ascent
    answer_loss = -1 * outputs.loss
    # Compute the Cross entropy loss for the question
    outputs = model(input_ids.to(device), labels=question_label.to(device), attention_mask=attention_mask.to(device))
    question_loss = outputs.loss
    return answer_loss, question_loss

# Forget Loss: IDK
def idk_loss(model, inputs):
    device = model.device
    # The first element of the data tuple is the target data
    forget_inputs = inputs[0]
    input_ids, answer_label, question_label, attention_mask = forget_inputs
    # Compute the Cross entropy loss for the answer - "I don't know"
    outputs = model(input_ids.to(device), labels=answer_label.to(device),
                    attention_mask=attention_mask.to(device))
    answer_loss = outputs.loss

    # Compute the Cross entropy loss for the question
    outputs = model(input_ids.to(device), labels=question_label.to(device),
                    attention_mask=attention_mask.to(device))
    question_loss = outputs.loss

    return answer_loss, question_loss

# Forget Loss: SIMNPO
def simnpo(model, inputs, beta = 2.5, gamma = 0.0):
    forget_inputs = inputs[0]
    device = model.device
    input_ids, answer_label, question_label, attention_mask = forget_inputs
    outputs = model(input_ids.to(device),labels=answer_label.to(device), attention_mask=attention_mask.to(device))
    loss_mask = answer_label != -100

    #Compute SIMNPO style loss for answer
    forget_loss = get_batch_loss(outputs.logits, answer_label.to(device)) / loss_mask.to(device).sum(-1) - gamma
    answer_loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta

    #Compute Cross entropy loss for question
    outputs = model(input_ids.to(device),labels=question_label.to(device), attention_mask=attention_mask.to(device))
    question_loss = outputs.loss

    return answer_loss.cpu(), question_loss.cpu()


# Regularization Loss: GD
def gd_loss(model, inputs):
    device = model.device
    retain_inputs = inputs[1]
    input_ids, alabels, qlabels, attention_mask = retain_inputs

    #here both labels are the same, since there is no question masking
    assert torch.all(alabels == qlabels)
    outputs = model(input_ids.to(device), labels=alabels.to(device),
                    attention_mask=attention_mask.to(device))
    loss = outputs.loss
    return loss


def get_batch_loss(output, labels):
    '''get the loss for each sequence in a batch **without averaging**. We need this for the SIMNPO loss, beacuse the losses are not just averaged'''
    #shift labels by 1 to the right
    shifted_labels = labels[..., 1:].contiguous()
    #shift output by 1 to the left
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss