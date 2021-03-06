## TODO:
# Modify this so that we get a dictionary of which gates to fix, we can then fix gates by layer - see which is truly causal?

# add args parser to make this easier to run

import torch
from torch import nn
from torch.nn import functional as F

import model
import data

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

import os


model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
data_location = 'subj_verb_generated_dataset_3'
save_location = 'subj_verb_generated_dataset_3_random_forget'
class_weights = None
#[forget, input, output, cell, hidden]
fix_gates=[True,False,False,False,False]
fix_values=[None,None,None,None,None]

if not os.path.exists(save_location):
    os.makedirs(save_location)

np.random.seed(1111)



# Load the pretrained model
with open(model_path, 'rb') as f:
    model = torch.load(f, map_location='cpu')

print('=== MODEL INFORMATION ===')
print(model)


print('\n\n=== MODEL PARAMATERS ===')
params = {}
for name, param in model.named_parameters():
    print(name)
    params[name] = param

model_values = {}
### LAYER 1 ###
# Current Timestep

encoder_weights = params['encoder.weight']
decoder_bias = params['decoder.bias']

for layer in range(model.rnn.num_layers):

    hidden_size = 650
    name = 'rnn.'

    w_ii_l0 = params[name+'weight_ih_l'+str(layer)][0:hidden_size]
    w_if_l0 = params[name+'weight_ih_l'+str(layer)][hidden_size:2*hidden_size]
    w_ig_l0 = params[name+'weight_ih_l'+str(layer)][2*hidden_size:3*hidden_size]
    w_io_l0 = params[name+'weight_ih_l'+str(layer)][3*hidden_size:4*hidden_size]

    b_ii_l0 = params[name+'bias_ih_l'+str(layer)][0:hidden_size]
    b_if_l0 = params[name+'bias_ih_l'+str(layer)][hidden_size:2*hidden_size]
    b_ig_l0 = params[name+'bias_ih_l'+str(layer)][2*hidden_size:3*hidden_size]
    b_io_l0 = params[name+'bias_ih_l'+str(layer)][3*hidden_size:4*hidden_size]

    input_vals = (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0)
    # Recurrent
    w_hi_l0 = params[name+'weight_hh_l'+str(layer)][0:hidden_size]
    w_hf_l0 = params[name+'weight_hh_l'+str(layer)][hidden_size:2*hidden_size]
    w_hg_l0 = params[name+'weight_hh_l'+str(layer)][2*hidden_size:3*hidden_size]
    w_ho_l0 = params[name+'weight_hh_l'+str(layer)][3*hidden_size:4*hidden_size]

    b_hi_l0 = params[name+'bias_hh_l'+str(layer)][0:hidden_size]
    b_hf_l0 = params[name+'bias_hh_l'+str(layer)][hidden_size:2*hidden_size]
    b_hg_l0 = params[name+'bias_hh_l'+str(layer)][2*hidden_size:3*hidden_size]
    b_ho_l0 = params[name+'bias_hh_l'+str(layer)][3*hidden_size:4*hidden_size]

    hidden_vals = (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0)

    model_values[layer] = (input_vals,hidden_vals)



def gated_forward(input, hidden, return_gates=False, fix_gates=[False,False,False,False,False], fix_values=[None,None,None,None,None]):
    """
    input and hidden are the same parameters passed into a forward for an RNN

    return_gates - True if you want a list of the value the gates took on during that time

    [forget, input, output, cell, hidden]
    fix_gates - this is list that will determine which gates will be fixed throughout the training

    fix_values - the list of values to which we will fix the gates
    """
    emb = model.drop(model.encoder(input))

    raw_output = emb
    new_hidden = []
    #raw_output, hidden = model.rnn(emb, hidden)
    raw_outputs = []
    outputs = []

    out_gates = []

    for l in range(model.rnn.num_layers):
        # print('LAYER ',l)
        (h0_l0, c0_l0) = hidden[l]
        i_vals, h_vals = model_values[l]

        (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0) = i_vals

        (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0) = h_vals

        gated_out = []

        f_gates, i_gates, o_gates, g_gates = [],[],[],[]

        fix_forget, fix_input, fix_output, fix_cell, fix_hidden = fix_gates

        val_forget, val_input, val_output, val_cell, val_hidden = fix_values

        for seq_i in range(len(raw_output)):
            inp = raw_output[seq_i]

            # forget gate
            f_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_if_l0)) + b_if_l0) + (torch.matmul(h0_l0,torch.t(w_hf_l0)) + b_hf_l0))

            if fix_forget:
                if val_forget is None:
                    f_g_l0 = torch.randn(f_g_l0.shape)



            # input gate
            i_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_ii_l0)) + b_ii_l0) + (torch.matmul(h0_l0,torch.t(w_hi_l0)) + b_hi_l0))

            if fix_input:
                if val_input is None:
                    i_g_l0 = torch.randn(i_g_l0.shape)

            # output gate
            o_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_io_l0)) + b_io_l0) + (torch.matmul(h0_l0, torch.t(w_ho_l0)) + b_ho_l0))

            if fix_output:
                if val_output is None:
                    o_g_l0 = torch.randn(o_g_l0.shape)

            # intermediate cell state
            c_tilde_l0 = torch.tanh((torch.matmul(inp,torch.t(w_ig_l0)) + b_ig_l0) + (torch.matmul(h0_l0, torch.t(w_hg_l0)) + b_hg_l0))


            # current cell state
            c0_l0 = f_g_l0 * c0_l0 + i_g_l0 * c_tilde_l0

            if fix_cell:
                if val_cell is None:
                    c0_l0 = torch.randn(c0_l0.shape)

            # hidden state
            h0_l0 = o_g_l0 * torch.tanh(c0_l0)

            if fix_hidden:
                if val_hidden is None:
                    h0_l0 = torch.randn(h0_l0.shape)


            new_h = (h0_l0,c0_l0)
            gated_out.append(h0_l0)

            f_gates.append(f_g_l0)
            i_gates.append(i_g_l0)
            o_gates.append(o_g_l0)
            g_gates.append(c0_l0)

        gates = (f_gates,i_gates,o_gates,g_gates)
        out_gates.append(gates)

        # print(h0_l0.shape)
        out = torch.stack(gated_out).reshape(len(gated_out),-1)
        raw_output = out

        new_hidden.append(new_h)
        raw_outputs.append(raw_output)
        if l != model.nlayers - 1:
            outputs.append(raw_output)
    hidden = new_hidden

    outputs.append(out)

    output = model.drop(out)

    # import pdb; pdb.set_trace()

    result = output

    decoded = model.decoder(result)

    if return_gates:
        return result, hidden, decoded, outputs, out_gates
    return result, hidden, decoded, outputs

def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, seq_len=1):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source,bsz,
fix_gates=[False,False,False,False,False], fix_values=[None,None,None,None,None]):
    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(bsz)

    data, targets = data_source[0:len(data_source)-1],data_source[1:]

    out,hidden,decoded, outputs, gates = gated_forward(data, hidden, return_gates=True,fix_gates=fix_gates,fix_values=fix_values)
    return gates, hidden, outputs

data_path = "../data/colorlessgreenRNNs/"


print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(data_path)

print('\n=== TESTING CORRECT MODEL ===')
input_path = "../data/subj_verb_generated/datasets/"+data_location+"_correct.txt"
print(input_path+'\n')
forget_gates_correct = {}
input_gates_correct = {}
output_gates_correct = {}
cell_states_correct = {}
hidden_states_correct = {}

plurality_labels_correct = {}

with open(input_path) as input_file:
    for line in input_file:
        agreement, singular, sentence = line.split('\t')
        agreement = int(agreement)
        singular = int(singular)

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        batch_size = 1
        input_data = batchify(tokenized_data,batch_size,False)

        gate_data, hiddens, outputs = evaluate(input_data,batch_size,fix_gates,fix_values)

        for lyr, gates in enumerate(gate_data):
            if lyr not in forget_gates_correct:
                forget_gates_correct[lyr] = []
            if lyr not in input_gates_correct:
                input_gates_correct[lyr] = []
            if lyr not in output_gates_correct:
                output_gates_correct[lyr] = []
            if lyr not in cell_states_correct:
                cell_states_correct[lyr] = []
            if lyr not in hidden_states_correct:
                hidden_states_correct[lyr] = []
            if lyr not in plurality_labels_correct:
                plurality_labels_correct[lyr] = []

            f_g_l0,i_g_l0,o_g_l0,c_tilde_l0 = gates
            hidden_values = outputs[lyr]
            for word in range(len(f_g_l0)):
                forget_gates_correct[lyr].append(f_g_l0[word].detach().numpy())
                input_gates_correct[lyr].append(i_g_l0[word].detach().numpy())
                output_gates_correct[lyr].append(o_g_l0[word].detach().numpy())
                cell_states_correct[lyr].append(c_tilde_l0[word].detach().numpy())
                hidden_states_correct[lyr].append(hidden_values[word].detach().numpy())
                plurality_labels_correct[lyr].append(singular)
            # uncomment to only see the first layer for testing
            # break

print('\n=== TESTING INCORRECT MODEL ===')
input_path = "../data/subj_verb_generated/datasets/"+data_location+"_incorrect.txt"
print(input_path+'\n')
forget_gates_incorrect = {}
input_gates_incorrect = {}
output_gates_incorrect = {}
cell_states_incorrect = {}
hidden_states_incorrect = {}

plurality_labels_incorrect = {}

number_of_words = 0
with open(input_path) as input_file:
    for line in input_file:
        agreement, singular, sentence = line.split('\t')

        number_of_words = len(sentence.split(' '))
        agreement = int(agreement)
        singular = int(singular)

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        batch_size = 1
        input_data = batchify(tokenized_data,batch_size,False)
        gate_data, hiddens, outputs = evaluate(input_data,batch_size,fix_gates,fix_values)
        # import pdb; pdb.set_trace()
        for lyr, gates in enumerate(gate_data):
            if lyr not in forget_gates_incorrect:
                forget_gates_incorrect[lyr] = []
            if lyr not in input_gates_incorrect:
                input_gates_incorrect[lyr] = []
            if lyr not in output_gates_incorrect:
                output_gates_incorrect[lyr] = []
            if lyr not in cell_states_incorrect:
                cell_states_incorrect[lyr] = []
            if lyr not in hidden_states_incorrect:
                hidden_states_incorrect[lyr] = []
            if lyr not in plurality_labels_incorrect:
                plurality_labels_incorrect[lyr] = []

            f_g_l0,i_g_l0,o_g_l0,c_tilde_l0 = gates
            hidden_values = outputs[lyr]
            for word in range(len(f_g_l0)):
                forget_gates_incorrect[lyr].append(f_g_l0[word].detach().numpy())
                input_gates_incorrect[lyr].append(i_g_l0[word].detach().numpy())
                output_gates_incorrect[lyr].append(o_g_l0[word].detach().numpy())
                cell_states_incorrect[lyr].append(c_tilde_l0[word].detach().numpy())
                hidden_states_incorrect[lyr].append(hidden_values[word].detach().numpy())
                plurality_labels_incorrect[lyr].append(singular)
            # uncomment to only see the first layer for testing
            # break

train_percent = 0.1
print('\n=== PLOTTING VIZ ===')
for lyr in range(2):
    """ FORGET GATE """
    train_number = int(train_percent*len(forget_gates_correct[lyr]))

    # correct
    training_indices = np.random.choice(range(len(forget_gates_correct[lyr])),train_number,replace=False)

    fgates_correct = np.array(forget_gates_correct[lyr])
    fgates_correct = fgates_correct[training_indices]
    fgates_correct = fgates_correct.reshape(train_number,-1)

    labels_correct = np.array(plurality_labels_correct[lyr])
    labels_correct = labels_correct[training_indices]

    # incorrect
    train_number = int(train_percent*len(forget_gates_incorrect[lyr]))
    training_indices = np.random.choice(range(len(forget_gates_incorrect[lyr])),train_number,replace=False)

    fgates_incorrect = np.array(forget_gates_incorrect[lyr])
    fgates_incorrect = fgates_incorrect[training_indices]
    fgates_incorrect = fgates_incorrect.reshape(train_number,-1)

    labels_incorrect = np.array(plurality_labels_incorrect[lyr])
    labels_incorrect = labels_incorrect[training_indices]

    # full
    fgates_full = np.vstack((fgates_correct,fgates_incorrect))
    labels_full = np.append(labels_correct,labels_incorrect)

    dc = sk.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights).fit(fgates_correct,labels_correct)

    fgates_correct = np.array(forget_gates_correct[lyr]).reshape(len(forget_gates_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    #correct
    running_mean_accuracies_correct = [0]*number_of_words

    for x_idx in range(len(fgates_correct)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_correct[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    fgates_incorrect = np.array(forget_gates_incorrect[lyr]).reshape(len(forget_gates_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])

    for x_idx in range(len(fgates_incorrect)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_incorrect[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Forget Gate Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_forget_gates.png')
    plt.close()

    """INPUT GATE"""
    train_number = int(train_percent*len(input_gates_correct[lyr]))

    # correct
    training_indices = np.random.choice(range(len(input_gates_correct[lyr])),train_number,replace=False)

    fgates_correct = np.array(input_gates_correct[lyr])
    fgates_correct = fgates_correct[training_indices]
    fgates_correct = fgates_correct.reshape(train_number,-1)

    labels_correct = np.array(plurality_labels_correct[lyr])
    labels_correct = labels_correct[training_indices]

    # incorrect
    train_number = int(train_percent*len(input_gates_incorrect[lyr]))
    training_indices = np.random.choice(range(len(input_gates_incorrect[lyr])),train_number,replace=False)

    fgates_incorrect = np.array(input_gates_incorrect[lyr])
    fgates_incorrect = fgates_incorrect[training_indices]
    fgates_incorrect = fgates_incorrect.reshape(train_number,-1)

    labels_incorrect = np.array(plurality_labels_incorrect[lyr])
    labels_incorrect = labels_incorrect[training_indices]

    # full
    fgates_full = np.vstack((fgates_correct,fgates_incorrect))
    labels_full = np.append(labels_correct,labels_incorrect)

    dc = sk.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights).fit(fgates_correct,labels_correct)

    fgates_correct = np.array(input_gates_correct[lyr]).reshape(len(input_gates_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    #correct
    running_mean_accuracies_correct = [0]*number_of_words

    for x_idx in range(len(fgates_correct)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_correct[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    fgates_incorrect = np.array(input_gates_incorrect[lyr]).reshape(len(input_gates_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])

    for x_idx in range(len(fgates_incorrect)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_incorrect[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Input Gate Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_input_gates.png')
    plt.close()

    """OUTPUT"""
    train_number = int(train_percent*len(output_gates_correct[lyr]))

    # correct
    training_indices = np.random.choice(range(len(output_gates_correct[lyr])),train_number,replace=False)

    fgates_correct = np.array(output_gates_correct[lyr])
    fgates_correct = fgates_correct[training_indices]
    fgates_correct = fgates_correct.reshape(train_number,-1)

    labels_correct = np.array(plurality_labels_correct[lyr])
    labels_correct = labels_correct[training_indices]

    # incorrect
    train_number = int(train_percent*len(output_gates_incorrect[lyr]))
    training_indices = np.random.choice(range(len(output_gates_incorrect[lyr])),train_number,replace=False)

    fgates_incorrect = np.array(output_gates_incorrect[lyr])
    fgates_incorrect = fgates_incorrect[training_indices]
    fgates_incorrect = fgates_incorrect.reshape(train_number,-1)

    labels_incorrect = np.array(plurality_labels_incorrect[lyr])
    labels_incorrect = labels_incorrect[training_indices]

    # full
    fgates_full = np.vstack((fgates_correct,fgates_incorrect))
    labels_full = np.append(labels_correct,labels_incorrect)

    dc = sk.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights).fit(fgates_correct,labels_correct)

    fgates_correct = np.array(output_gates_correct[lyr]).reshape(len(output_gates_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    #correct
    running_mean_accuracies_correct = [0]*number_of_words

    for x_idx in range(len(fgates_correct)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_correct[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    fgates_incorrect = np.array(output_gates_incorrect[lyr]).reshape(len(output_gates_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])

    for x_idx in range(len(fgates_incorrect)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_incorrect[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Output Gate Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_output_gates.png')
    plt.close()

    """CELL STATE"""
    train_number = int(train_percent*len(cell_states_correct[lyr]))

    # correct
    training_indices = np.random.choice(range(len(cell_states_correct[lyr])),train_number,replace=False)

    fgates_correct = np.array(cell_states_correct[lyr])
    fgates_correct = fgates_correct[training_indices]
    fgates_correct = fgates_correct.reshape(train_number,-1)

    labels_correct = np.array(plurality_labels_correct[lyr])
    labels_correct = labels_correct[training_indices]

    # incorrect
    train_number = int(train_percent*len(cell_states_incorrect[lyr]))
    training_indices = np.random.choice(range(len(cell_states_incorrect[lyr])),train_number,replace=False)

    fgates_incorrect = np.array(cell_states_incorrect[lyr])
    fgates_incorrect = fgates_incorrect[training_indices]
    fgates_incorrect = fgates_incorrect.reshape(train_number,-1)

    labels_incorrect = np.array(plurality_labels_incorrect[lyr])
    labels_incorrect = labels_incorrect[training_indices]

    # full
    fgates_full = np.vstack((fgates_correct,fgates_incorrect))
    labels_full = np.append(labels_correct,labels_incorrect)

    dc = sk.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights).fit(fgates_correct,labels_correct)

    fgates_correct = np.array(cell_states_correct[lyr]).reshape(len(cell_states_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    #correct
    running_mean_accuracies_correct = [0]*number_of_words

    for x_idx in range(len(fgates_correct)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_correct[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    fgates_incorrect = np.array(cell_states_incorrect[lyr]).reshape(len(cell_states_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])

    for x_idx in range(len(fgates_incorrect)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_incorrect[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Cell State Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_cell_states.png')
    plt.close()

    """HIDDEN STATE"""
    train_number = int(train_percent*len(hidden_states_correct[lyr]))

    # correct
    training_indices = np.random.choice(range(len(hidden_states_correct[lyr])),train_number,replace=False)

    fgates_correct = np.array(hidden_states_correct[lyr])
    fgates_correct = fgates_correct[training_indices]
    fgates_correct = fgates_correct.reshape(train_number,-1)

    labels_correct = np.array(plurality_labels_correct[lyr])
    labels_correct = labels_correct[training_indices]

    # incorrect
    train_number = int(train_percent*len(hidden_states_incorrect[lyr]))
    training_indices = np.random.choice(range(len(hidden_states_incorrect[lyr])),train_number,replace=False)

    fgates_incorrect = np.array(hidden_states_incorrect[lyr])
    fgates_incorrect = fgates_incorrect[training_indices]
    fgates_incorrect = fgates_incorrect.reshape(train_number,-1)

    labels_incorrect = np.array(plurality_labels_incorrect[lyr])
    labels_incorrect = labels_incorrect[training_indices]

    # full
    fgates_full = np.vstack((fgates_correct,fgates_incorrect))
    labels_full = np.append(labels_correct,labels_incorrect)

    dc = sk.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights).fit(fgates_correct,labels_correct)

    fgates_correct = np.array(hidden_states_correct[lyr]).reshape(len(hidden_states_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    #correct
    running_mean_accuracies_correct = [0]*number_of_words

    for x_idx in range(len(fgates_correct)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_correct[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    fgates_incorrect = np.array(hidden_states_incorrect[lyr]).reshape(len(hidden_states_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])

    for x_idx in range(len(fgates_incorrect)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates_incorrect[x_idx].reshape(1,-1)
        accurate = int(int(dc.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Hidden State Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_hidden_states.png')
    plt.close()
# END OF FILE
