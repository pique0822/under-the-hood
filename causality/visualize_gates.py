import torch
from torch import nn
from torch.nn import functional as F

import scipy.io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import model
import data

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'



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
            g_gates.append(c_tilde_l0)

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
        return result, hidden, decoded, out_gates
    return result, hidden, decoded,

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

def evaluate(data_source,bsz):
    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(bsz)

    data, targets = data_source[0:len(data_source)-1],data_source[1:]

    out,hidden,decoded,gates = gated_forward(data, hidden, return_gates=True)

    import pdb; pdb.set_trace()
    
    return gates

data_path = "../data/colorlessgreenRNNs/"
input_path = "../data/subj_verb_generated/subj_verb_generated_dataset.txt"

print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(data_path)

print('\n=== TESTING MODEL ===')

agreement_singular = {}
agreement_plural = {}
opposite_singular = {}
opposite_plural = {}

with open(input_path) as input_file:
    for line in input_file:
        agreement, singular, sentence = line.split('\t')
        agreement = int(agreement)
        singular = int(singular)

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        batch_size = 1
        input_data = batchify(tokenized_data,batch_size,False)
        gate_data = evaluate(input_data,batch_size)

        for lyr, gates in enumerate(gate_data):
            if lyr not in agreement_singular:
                agreement_singular[lyr] = []
            if lyr not in agreement_plural:
                agreement_plural[lyr] = []
            if lyr not in opposite_singular:
                opposite_singular[lyr] = []
            if lyr not in opposite_plural:
                opposite_plural[lyr] = []

            f_g_l0,i_g_l0,o_g_l0,c_tilde_l0 = gates

            for word in range(len(f_g_l0)):
                if agreement == 1 and singular == 1:
                    agreement_singular[lyr].append(f_g_l0[word].detach().numpy())
                elif agreement == 1 and singular == 0:
                    agreement_plural[lyr].append(f_g_l0[word].detach().numpy())
                elif agreement == 0 and singular == 1:
                    opposite_singular[lyr].append(f_g_l0[word].detach().numpy())
                elif agreement == 0 and singular == 0:
                    opposite_plural[lyr].append(f_g_l0[word].detach().numpy())
            # uncomment to only see the first layer for testing
            # break
        # uncomment for testing
        # break



save_location = ''
print('\n=== PLOTTING VIZ ===')
for lyr in range(2):
    print('PCA LAYER ',lyr)
    ag_sing = agreement_singular[lyr]
    ag_plu = agreement_plural[lyr]
    op_sing = opposite_singular[lyr]
    op_plu = opposite_plural[lyr]

    all_forgets = ag_sing.copy()
    all_forgets.extend(ag_plu)
    all_forgets.extend(op_sing)
    all_forgets.extend(op_plu)

    ag_sing = np.array(ag_sing).reshape(len(ag_sing),-1)
    ag_plu = np.array(ag_plu).reshape(len(ag_plu),-1)
    op_sing = np.array(op_sing).reshape(len(op_sing),-1)
    op_plu = np.array(op_plu).reshape(len(op_plu),-1)

    all_forgets = np.array(all_forgets).reshape(len(all_forgets),-1)
    forget_emb = PCA(n_components=2).fit_transform(all_forgets)

    x,y = forget_emb[0]
    plt.plot(x,y,'co',label='Agreement Singular')

    x,y = forget_emb[len(ag_sing)]
    plt.plot(x,y,'bo',label='Agreement Plural')

    x,y = forget_emb[len(ag_sing) + len(ag_plu)]
    plt.plot(x,y,'mo',label='Opposite Singular')

    x,y = forget_emb[len(ag_sing) + len(ag_plu) + len(op_sing)]
    plt.plot(x,y,'ro',label='Opposite Plural')
    for i in range(len(forget_emb)):
        x,y = forget_emb[i]

        if i < len(ag_sing):
            plt.plot(x,y,'co',alpha=1)
        elif i < len(ag_sing)+len(ag_plu):
            plt.plot(x,y,'bo',alpha=1)
        elif i < len(ag_sing)+len(ag_plu)+len(op_sing):
            plt.plot(x,y,'mo',alpha=1)
        else:
            plt.plot(x,y,'ro',alpha=1)

    plt.title('PCA(n=2) of Forget Gate Representations in Layer '+str(lyr))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(save_location+'lyr_'+str(lyr)+'_viz.png')
    plt.close()




# END OF FILE
