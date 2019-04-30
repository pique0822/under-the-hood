import torch
from torch import nn
from torch.nn import functional as F

import model
import data

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import pandas as pd

import os
import utils

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
save_location = '2D_viz'
class_weights = 'balanced'

load_files = False

if not os.path.exists(save_location):
    os.makedirs(save_location)

np.random.seed(42)

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


ambiguous_cols = ['Start','Noun','Ambiguous verb']
whowas_ambiguous_cols = ['Start','Noun','Unreduced content','Ambiguous verb']
unambiguous_cols = ['Start','Noun','Unambiguous verb']
whowas_unambiguous_cols = ['Start','Noun','Unreduced content','Unambiguous verb']

ambiguous_hiddens = []
whowas_ambiguous_hiddens = []
unambiguous_hiddens = []
whowas_unambiguous_hiddens = []

all_hiddens = []
# will store ambiguous vs not ambiguous
all_labels = []


if not load_files:
    print('\n=== TESTING MODEL ===')
    for df_idx in range(len(df)):
        print('')
        row = df.iloc[df_idx]

        hidden = model.init_hidden(1)

        ambiguous_sentence = ""
        for column in ambiguous_cols:
            ambiguous_sentence += ' '+row[column]
            ambiguous_sentence = ambiguous_sentence.lstrip().strip()

        whowas_ambiguous_sentence = ""
        for column in whowas_ambiguous_cols:
            whowas_ambiguous_sentence += ' '+row[column]
            whowas_ambiguous_sentence = whowas_ambiguous_sentence.lstrip().strip()

        unambiguous_sentence = ""
        for column in unambiguous_cols:
            unambiguous_sentence += ' '+row[column]
            unambiguous_sentence = unambiguous_sentence.lstrip().strip()

        whowas_unambiguous_sentence = ""
        for column in whowas_unambiguous_cols:
            whowas_unambiguous_sentence += ' '+row[column]
            whowas_unambiguous_sentence = whowas_unambiguous_sentence.lstrip().strip()

        print(ambiguous_sentence)

        tokenized_ambiguous = corpus.safe_tokenize_sentence(ambiguous_sentence.strip())
        tokenized_whowas_ambiguous = corpus.safe_tokenize_sentence(whowas_ambiguous_sentence.strip())

        tokenized_unambiguous = corpus.safe_tokenize_sentence(unambiguous_sentence.strip())
        tokenized_whowas_unambiguous = corpus.safe_tokenize_sentence(whowas_unambiguous_sentence.strip())


        input_data = batchify(tokenized_ambiguous,1,False)
        gate_data, hidden, outputs = evaluate(input_data, hidden)

        last_hidden = hidden[1][0].detach().numpy()

        all_hiddens.append(last_hidden)
        ambiguous_hiddens.append(last_hidden)
        all_labels.append(1)


        input_data = batchify(tokenized_whowas_ambiguous,1,False)
        gate_data, hidden, outputs = evaluate(input_data, hidden)

        last_hidden = hidden[1][0].detach().numpy()

        all_hiddens.append(last_hidden)
        whowas_ambiguous_hiddens.append(last_hidden)
        all_labels.append(0)

        input_data = batchify(tokenized_unambiguous,1,False)
        gate_data, hidden, outputs = evaluate(input_data, hidden)

        last_hidden = hidden[1][0].detach().numpy()

        all_hiddens.append(last_hidden)
        unambiguous_hiddens.append(last_hidden)
        all_labels.append(0)

        input_data = batchify(tokenized_whowas_unambiguous,1,False)
        gate_data, hidden, outputs = evaluate(input_data, hidden)

        last_hidden = hidden[1][0].detach().numpy()

        all_hiddens.append(last_hidden)
        whowas_unambiguous_hiddens.append(last_hidden)
        all_labels.append(0)

print('\n=== PLOTTING VIZ ===')

all_hiddens = np.array(all_hiddens)
all_hiddens = all_hiddens.reshape(len(all_hiddens),-1)

all_labels = np.array(all_labels)

ambiguous_hiddens = np.array(ambiguous_hiddens)
ambiguous_hiddens = ambiguous_hiddens.reshape(len(ambiguous_hiddens),-1)

whowas_ambiguous_hiddens = np.array(whowas_ambiguous_hiddens)
whowas_ambiguous_hiddens = whowas_ambiguous_hiddens.reshape(len(whowas_ambiguous_hiddens),-1)

unambiguous_hiddens = np.array(unambiguous_hiddens)
unambiguous_hiddens = unambiguous_hiddens.reshape(len(unambiguous_hiddens),-1)

whowas_unambiguous_hiddens = np.array(whowas_unambiguous_hiddens)
whowas_unambiguous_hiddens = whowas_unambiguous_hiddens.reshape(len(whowas_unambiguous_hiddens),-1)

train_percent = 0.1
train_number = int(train_percent*len(all_hiddens))
# correct
training_indices = np.random.choice(range(len(all_hiddens)),train_number,replace=False)
# logistic regression

reg_all = sk.LogisticRegression(solver='lbfgs',class_weight='balanced').fit(all_hiddens[training_indices],all_labels[training_indices])

label = np.ones(len(ambiguous_hiddens))
print('Ambiguous Sentence Prediction Accuracy',reg_all.score(ambiguous_hiddens,label))

label = np.zeros(len(whowas_ambiguous_hiddens))
print('Who Was Ambiguous Sentence Prediction Accuracy',reg_all.score(whowas_ambiguous_hiddens,label))

label = np.zeros(len(unambiguous_hiddens))
print('Unambiguous Sentence Prediction Accuracy',reg_all.score(unambiguous_hiddens,label))

label = np.zeros(len(whowas_unambiguous_hiddens))
print('Who Was Unambiguous Sentence Prediction Accuracy',reg_all.score(whowas_unambiguous_hiddens,label))

mean_coef = reg_all.coef_[0].mean()
std_coef = reg_all.coef_[0].std()
significant_coef_indices = np.where(np.abs(reg_all.coef_[0]) > mean_coef + 3*std_coef)[0]

# print('Significant Units',significant_coef_indices)

coef_count = {}

print('=== 100 Experiment Significant ===')
num_runs = 0
for exper_idx in range(1000):

    train_percent = 0.1
    train_number = int(train_percent*len(all_hiddens))
    # correct
    training_indices = np.random.choice(range(len(all_hiddens)),train_number,replace=False)
    # logistic regression

    if not sum(all_labels[training_indices]) >= 1:
        continue

    num_runs += 1

    reg = sk.LogisticRegression(solver='lbfgs',class_weight='balanced').fit(all_hiddens[training_indices],all_labels[training_indices])

    mean_coef = reg.coef_[0].mean()
    std_coef = reg.coef_[0].std()
    significant_coef_indices = np.where(np.abs(reg.coef_[0]) > mean_coef + 3*std_coef)[0]

    for coef_idx in significant_coef_indices:
        if coef_idx in coef_count:
            coef_count[coef_idx] += 1
        else:
            coef_count[coef_idx] = 1

coef_count_values = np.array(list(coef_count.values()))
mean_coef_count = coef_count_values.mean()
std_coef_count = coef_count_values.std()

true_significance = mean_coef_count + 3*std_coef_count

significant_coef_indices = []


for index,count in coef_count.items():
    if count > true_significance:
        significant_coef_indices.append(index)
print('True Significant Units',significant_coef_indices)


significant_coef_indices = [410, 499, 613, 29, 157, 608]

modified_ambiguous_hiddens = ambiguous_hiddens.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    for row in range(modified_ambiguous_hiddens.shape[0]):
        modified_ambiguous_hiddens[row][coef_idx] = 0

modified_whowas_ambiguous_hiddens = whowas_ambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_whowas_ambiguous_hiddens.shape[0]):
        modified_whowas_ambiguous_hiddens[row][idx] = 0

modified_unambiguous_hiddens = unambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_unambiguous_hiddens.shape[0]):
        modified_unambiguous_hiddens[row][idx] = 0

modified_whowas_unambiguous_hiddens = whowas_unambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_whowas_unambiguous_hiddens.shape[0]):
        modified_whowas_unambiguous_hiddens[row][idx] = 0

print('\n\n ZEROS \n\n')
label = np.ones(len(modified_ambiguous_hiddens))
print('MOD Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_ambiguous_hiddens,label))

label = np.zeros(len(modified_whowas_ambiguous_hiddens))
print('MOD Who Was Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_whowas_ambiguous_hiddens,label))

label = np.zeros(len(modified_unambiguous_hiddens))
print('MOD Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_unambiguous_hiddens,label))

label = np.zeros(len(modified_whowas_unambiguous_hiddens))
print('MOD Who Was Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_whowas_unambiguous_hiddens,label))


modified_ambiguous_hiddens = ambiguous_hiddens.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    for row in range(modified_ambiguous_hiddens.shape[0]):
        modified_ambiguous_hiddens[row][coef_idx] = 1

modified_whowas_ambiguous_hiddens = whowas_ambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_whowas_ambiguous_hiddens.shape[0]):
        modified_whowas_ambiguous_hiddens[row][idx] = 1

modified_unambiguous_hiddens = unambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_unambiguous_hiddens.shape[0]):
        modified_unambiguous_hiddens[row][idx] = 1

modified_whowas_unambiguous_hiddens = whowas_unambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_whowas_unambiguous_hiddens.shape[0]):
        modified_whowas_unambiguous_hiddens[row][idx] = 1

print('\n\n ONES \n\n')
label = np.ones(len(modified_ambiguous_hiddens))
print('MOD Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_ambiguous_hiddens,label))

label = np.zeros(len(modified_whowas_ambiguous_hiddens))
print('MOD Who Was Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_whowas_ambiguous_hiddens,label))

label = np.zeros(len(modified_unambiguous_hiddens))
print('MOD Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_unambiguous_hiddens,label))

label = np.zeros(len(modified_whowas_unambiguous_hiddens))
print('MOD Who Was Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_whowas_unambiguous_hiddens,label))


modified_ambiguous_hiddens = ambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_ambiguous_hiddens.shape[0]):
        modified_ambiguous_hiddens[row][idx] = -1

modified_whowas_ambiguous_hiddens = whowas_ambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_whowas_ambiguous_hiddens.shape[0]):
        modified_whowas_ambiguous_hiddens[row][idx] = -1

modified_unambiguous_hiddens = unambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_unambiguous_hiddens.shape[0]):
        modified_unambiguous_hiddens[row][idx] = -1

modified_whowas_unambiguous_hiddens = whowas_unambiguous_hiddens.copy()
for idx in significant_coef_indices:
    for row in range(modified_whowas_unambiguous_hiddens.shape[0]):
        modified_whowas_unambiguous_hiddens[row][idx] = -1

print('\n\n NEG 1 \n\n')
label = np.ones(len(modified_ambiguous_hiddens))
print('MOD Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_ambiguous_hiddens,label))

label = np.zeros(len(modified_whowas_ambiguous_hiddens))
print('MOD Who Was Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_whowas_ambiguous_hiddens,label))

label = np.zeros(len(modified_unambiguous_hiddens))
print('MOD Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_unambiguous_hiddens,label))

label = np.zeros(len(modified_whowas_unambiguous_hiddens))
print('MOD Who Was Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_whowas_unambiguous_hiddens,label))


modified_ambiguous_hiddens = ambiguous_hiddens.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    value = -1
    for row in range(modified_ambiguous_hiddens.shape[0]):
        modified_ambiguous_hiddens[row][coef_idx] = value
modified_unambiguous_hiddens = unambiguous_hiddens.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    value = 1
    for row in range(modified_unambiguous_hiddens.shape[0]):
        modified_unambiguous_hiddens[row][coef_idx] = value

print('\n\n SURGICAL \n\n')
label = np.ones(len(modified_ambiguous_hiddens))
print('MOD Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_ambiguous_hiddens,label))

label = np.zeros(len(modified_unambiguous_hiddens))
print('MOD Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_unambiguous_hiddens,label))

unambiguous_significant = unambiguous_hiddens[:,significant_coef_indices]

print('UNAM MEAN VALUE',unambiguous_significant.mean(0))

ambiguous_significant = ambiguous_hiddens[:,significant_coef_indices]
print('AM MEAN VALUE',ambiguous_significant.mean(0))


print('\n\n TRAINING FROM START \n\n')

modified_ambiguous_hiddens = ambiguous_hiddens.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    value = -1
    for row in range(modified_ambiguous_hiddens.shape[0]):
        modified_ambiguous_hiddens[row][coef_idx] = value
modified_unambiguous_hiddens = unambiguous_hiddens.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    value = 1
    for row in range(modified_unambiguous_hiddens.shape[0]):
        modified_unambiguous_hiddens[row][coef_idx] = value

modified_all_hiddens = whowas_ambiguous_hiddens.copy()
modified_all_hiddens = np.vstack((modified_all_hiddens,whowas_unambiguous_hiddens))
modified_all_hiddens = np.vstack((modified_all_hiddens,modified_unambiguous_hiddens))
modified_all_hiddens = np.vstack((modified_all_hiddens,modified_ambiguous_hiddens))
modified_all_labels = np.zeros(len(whowas_ambiguous_hiddens) + len(whowas_unambiguous_hiddens)+ len(modified_unambiguous_hiddens))
modified_all_labels = np.concatenate((modified_all_labels, np.ones(len(modified_ambiguous_hiddens))))


train_percent = 0.25
train_number = int(train_percent*len(all_hiddens))
# correct
training_indices = np.random.choice(range(len(all_hiddens)),train_number,replace=False)
reg = sk.LogisticRegression(solver='lbfgs',class_weight='balanced').fit(modified_all_hiddens[training_indices],modified_all_labels[training_indices])

label = np.ones(len(modified_ambiguous_hiddens))
print('MOD Ambiguous Sentence Prediction Accuracy',reg_all.score(modified_ambiguous_hiddens,label))

label = np.zeros(len(whowas_ambiguous_hiddens))
print('MOD Who Was Ambiguous Sentence Prediction Accuracy',reg_all.score(whowas_ambiguous_hiddens,label))

label = np.zeros(len(modified_unambiguous_hiddens))
print('MOD Unambiguous Sentence Prediction Accuracy',reg_all.score(modified_unambiguous_hiddens,label))

label = np.zeros(len(whowas_unambiguous_hiddens))
print('MOD Who Was Unambiguous Sentence Prediction Accuracy',reg_all.score(whowas_unambiguous_hiddens,label))

# import pdb; pdb.set_trace()

# from simple observations, it seems that these act as flags in the sense that setting these vaulues to -1 will make the sentence think tht they are ambiguous while setting them to 1 will make them think it is unambiguous

# import pdb; pdb.set_trace()

significant_hiddens = all_hiddens[:,significant_coef_indices]
# import pdb; pdb.set_trace()

pca = PCA(n_components = 2)
pca.fit(all_hiddens)
print('Explained Variance::',sum(pca.explained_variance_ratio_))

emb_vecs = pca.transform(ambiguous_hiddens)
col = '#1abc9c'
x,y = emb_vecs[0]
plt.scatter(x,y,c=col, label='Ambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(whowas_ambiguous_hiddens)
x,y = emb_vecs[0]
col = '#3498db'
plt.scatter(x,y,c=col, label='Who ambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(unambiguous_hiddens)
x,y = emb_vecs[0]
col = '#f1c40f'
plt.scatter(x,y,c=col,label='Unambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(whowas_unambiguous_hiddens)
col = '#e74c3c'
x,y = emb_vecs[0]
plt.scatter(x,y,c=col,label='Who unambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

plt.title('PCA of Layer 2 Hidden State, Explained Var '+str(round(sum(pca.explained_variance_ratio_)*100,1))+'%')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


pca = PCA(n_components = 2)
pca.fit(significant_hiddens)
print('Explained Variance::',sum(pca.explained_variance_ratio_))

emb_vecs = pca.transform(ambiguous_hiddens[:,significant_coef_indices])
col = '#1abc9c'
x,y = emb_vecs[0]
plt.scatter(x,y,c=col, label='Ambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(whowas_ambiguous_hiddens[:,significant_coef_indices])
x,y = emb_vecs[0]
col = '#3498db'
plt.scatter(x,y,c=col, label='Who ambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(unambiguous_hiddens[:,significant_coef_indices])
x,y = emb_vecs[0]
col = '#f1c40f'
plt.scatter(x,y,c=col,label='Unambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(whowas_unambiguous_hiddens[:,significant_coef_indices])
col = '#e74c3c'
x,y = emb_vecs[0]
plt.scatter(x,y,c=col,label='Who unambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

plt.title('Significant Units PCA of Layer 2 Hidden State, Explained Var '+str(round(sum(pca.explained_variance_ratio_)*100,1))+'%')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


pca = PCA(n_components = 2)
pca.fit(modified_all_hiddens)
print('Explained Variance::',sum(pca.explained_variance_ratio_))

emb_vecs = pca.transform(modified_ambiguous_hiddens)
col = '#1abc9c'
x,y = emb_vecs[0]
plt.scatter(x,y,c=col, label='Ambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(modified_whowas_ambiguous_hiddens)
x,y = emb_vecs[0]
col = '#3498db'
plt.scatter(x,y,c=col, label='Who ambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(modified_unambiguous_hiddens)
x,y = emb_vecs[0]
col = '#f1c40f'
plt.scatter(x,y,c=col,label='Unambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

emb_vecs = pca.transform(modified_whowas_unambiguous_hiddens)
col = '#e74c3c'
x,y = emb_vecs[0]
plt.scatter(x,y,c=col,label='Who unambiguous')
for vec in emb_vecs:
    x,y = vec
    plt.scatter(x,y,c=col)

plt.title('Modified PCA of Layer 2 Hidden State, Explained Var '+str(round(sum(pca.explained_variance_ratio_)*100,1))+'%')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

tsne = TSNE(n_components = 2, perplexity = 5)
emb_vecs = tsne.fit_transform(all_hiddens)

for idx, vec in enumerate(emb_vecs):
    x,y = vec
    if all_labels[idx] == 1:
        col = '#1abc9c'
    else:
        col = '#e74c3c'
    plt.scatter(x,y,c=col)


plt.title('TSNE')
plt.show()

tsne = TSNE(n_components = 2, perplexity = 5)
emb_vecs = tsne.fit_transform(all_hiddens[:,significant_coef_indices])

for idx, vec in enumerate(emb_vecs):
    x,y = vec
    if all_labels[idx] == 1:
        col = '#1abc9c'
    else:
        col = '#e74c3c'
    plt.scatter(x,y,c=col)


plt.title('TSNE')
plt.show()


tsne = TSNE(n_components = 2, perplexity = 5)
emb_vecs = tsne.fit_transform(modified_all_hiddens)

for idx, vec in enumerate(emb_vecs):
    x,y = vec
    if modified_all_labels[idx] == 1:
        col = '#1abc9c'
    else:
        col = '#e74c3c'
    plt.scatter(x,y,c=col)


plt.title('TSNE')
plt.show()

pred_labels = KMeans(n_clusters = 2).fit_predict(all_hiddens)
accuracy = sum(pred_labels == all_labels)/len(pred_labels)

print('All Hidden Units Clustering 2',accuracy)

significant_hiddens = all_hiddens[:,significant_coef_indices]

pred_labels = KMeans(n_clusters = 2).fit_predict(significant_hiddens)
accuracy = sum(pred_labels == all_labels)/len(pred_labels)

print('Significant Hidden Units Clustering 2',accuracy)

# import pdb; pdb.set_trace()


# import pdb; pdb.set_trace()



# END OF FILE
