import torch
from torch import nn
from torch.nn import functional as F

import model
import data

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

import pandas as pd

import os
import utils

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
save_location = 'DCs'
class_weights = 'balanced'

load_files = True

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

#
def gated_forward(input, hidden, return_gates=False):
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
        for seq_i in range(len(raw_output)):
            inp = raw_output[seq_i]

            # forget gate
            f_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_if_l0)) + b_if_l0) + (torch.matmul(h0_l0,torch.t(w_hf_l0)) + b_hf_l0))


            # input gate
            i_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_ii_l0)) + b_ii_l0) + (torch.matmul(h0_l0,torch.t(w_hi_l0)) + b_hi_l0))

            # output gate
            o_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_io_l0)) + b_io_l0) + (torch.matmul(h0_l0, torch.t(w_ho_l0)) + b_ho_l0))


            # intermediate cell state
            c_tilde_l0 = torch.tanh((torch.matmul(inp,torch.t(w_ig_l0)) + b_ig_l0) + (torch.matmul(h0_l0, torch.t(w_hg_l0)) + b_hg_l0))


            # current cell state
            c0_l0 = f_g_l0 * c0_l0 + i_g_l0 * c_tilde_l0

            # hidden state
            h0_l0 = o_g_l0 * torch.tanh(c0_l0)


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

def evaluate(data_source, hidden):
    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)

    data, targets = data_source[0:len(data_source)-1],data_source[1:]

    out,hidden,decoded, outputs, gates = gated_forward(data, hidden, return_gates=True)
    return gates, hidden, outputs

data_path = "../data/colorlessgreenRNNs/"


print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(data_path)


df = pd.read_csv('data/verb-ambiguity-with-intervening-phrase.csv')

forget_gates = {}
input_gates = {}
output_gates = {}
cell_states = {}
hidden_states = {}

plurality_labels = {}

columns_for_sentence = ['Start','Noun','Ambiguous verb','RC contents','Disambiguator','End']


def scoring_function(y_pred,y_true,round=5):
    # this is the absolute error difference
    return np.round_((y_pred - y_true)/y_true,round)

if not load_files:
    print('=== TESTING MODEL ===')
    for df_idx in range(len(df)):
        print('')
        row = df.iloc[df_idx]

        hidden = model.init_hidden(1)
        parsed_sentence = ""

        main_verb = row['Disambiguator'].strip().lstrip()
        for column in columns_for_sentence:
            print(parsed_sentence)

            label = 0
            text = row[column].strip().lstrip()

            split_text = text.split(' ')

            tokenized_data = corpus.safe_tokenize_sentence(text.strip())



            input_data = batchify(tokenized_data,1,False)
            gate_data, hidden, outputs = evaluate(input_data, hidden)

            for lyr, gates in enumerate(gate_data):
                current_sentence = parsed_sentence

                if lyr not in forget_gates:
                    forget_gates[lyr] = {}
                    for sub_col in columns_for_sentence:
                        forget_gates[lyr][sub_col] = []
                if lyr not in input_gates:
                    input_gates[lyr] = {}
                    for sub_col in columns_for_sentence:
                        input_gates[lyr][sub_col] = []
                if lyr not in output_gates:
                    output_gates[lyr] = {}
                    for sub_col in columns_for_sentence:
                        output_gates[lyr][sub_col] = []
                if lyr not in cell_states:
                    cell_states[lyr] = {}
                    for sub_col in columns_for_sentence:
                        cell_states[lyr][sub_col] = []
                if lyr not in hidden_states:
                    hidden_states[lyr] = {}
                    for sub_col in columns_for_sentence:
                        hidden_states[lyr][sub_col] = []
                if lyr not in plurality_labels:
                    plurality_labels[lyr] = {}
                    for sub_col in columns_for_sentence:
                        plurality_labels[lyr][sub_col] = []

                f_g_l0,i_g_l0,o_g_l0,c_tilde_l0 = gates
                hidden_values = outputs[lyr]

                avg_f_g_l0,avg_i_g_l0,avg_o_g_l0,avg_c_tilde_l0 = f_g_l0,i_g_l0,o_g_l0,c_tilde_l0
                for word in range(len(f_g_l0)):
                    forget_gates[lyr][column].append(f_g_l0[word].detach().numpy())
                    input_gates[lyr][column].append(i_g_l0[word].detach().numpy())
                    output_gates[lyr][column].append(o_g_l0[word].detach().numpy())
                    cell_states[lyr][column].append(c_tilde_l0[word].detach().numpy())
                    hidden_states[lyr][column].append(hidden_values[word].detach().numpy())

                    current_sentence += ' ' + split_text[word]
                    current_sentence = current_sentence.lstrip()

                    largest_prob_idx, cont_surps, memoize = utils.most_likely_continuation(model,corpus.dictionary,current_sentence,['.',main_verb],return_prob=False)

                    period_surp, verb_surp = cont_surps

                    # this should be when the sentence is at the period

                    plurality_labels[lyr][column].append(period_surp - verb_surp)
            parsed_sentence = current_sentence
                # uncomment to only see the first layer for testing
                # break

train_percent = .1

accuracies_matrix = np.zeros((len(columns_for_sentence),len(columns_for_sentence)))
print('\n=== PLOTTING VIZ ===')
for lyr in [1]:
    for cidx,training_column in enumerate(columns_for_sentence):
        save_name = ''.join(training_column.split(' '))
        if not load_files:
            unspecified_gates = hidden_states[lyr][training_column]
            unspecified_labels = plurality_labels[lyr][training_column]

            np.save('data/saved_arrays/'+save_name+'_hidden.npy', unspecified_gates)
            np.save('data/saved_arrays/'+save_name+'_labels.npy', unspecified_labels)
        else:
            unspecified_gates = np.load('data/saved_arrays/'+save_name+'_hidden.npy')
            unspecified_labels = np.load('data/saved_arrays/'+save_name+'_labels.npy')



        train_number = int(train_percent*len(unspecified_gates))

        # correct
        training_indices = np.random.choice(range(len(unspecified_gates)),train_number,replace=False)

        fgates = np.array(unspecified_gates)
        fgates = fgates[training_indices]
        fgates = fgates.reshape(train_number,-1)

        labels = np.array(unspecified_labels)
        labels = labels[training_indices]

        dc = sk.LinearRegression().fit(fgates,labels)

        accuracy = 0
        for ridx,test_column in enumerate(columns_for_sentence):
            if not load_files:
                gates = np.array(hidden_states[lyr][test_column]).reshape(len(hidden_states[lyr][test_column]),-1)

                labels = np.array(plurality_labels[lyr][test_column])
            else:
                test_name = ''.join(test_column.split(' '))
                gates = np.load('data/saved_arrays/'+test_name+'_hidden.npy')
                labels = np.load('data/saved_arrays/'+test_name+'_labels.npy')


            pred_rs = dc.predict(gates)

            scores = scoring_function(pred_rs,labels,1)
            accuracy = np.round(np.sum(scores)/len(scores),1)
            print(accuracy)
            # import pdb; pdb.set_trace()
            accuracies_matrix[ridx][cidx] = accuracy

    plt.imshow(accuracies_matrix, cmap='winter', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title('Percent Error')
    plt.xlabel('Trained On')
    plt.ylabel('Tested On')

    for i in range(accuracies_matrix.shape[0]):
        for j in range(accuracies_matrix.shape[1]):
            text = plt.text(j, i, str(accuracies_matrix[i, j]) + '%',
                       ha="center", va="center", color="w")

    plt.savefig(save_location+'/lyr_'+str(lyr)+'_forget_gates.png')
    plt.show()
    plt.close()

# END OF FILE
