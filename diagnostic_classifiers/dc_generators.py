import torch
from torch import nn
from torch.nn import functional as F

import model
import data

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
data_location = 'subj_verb_generated_dataset_3'
save_location = data_location

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

def evaluate(data_source,bsz):
    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(bsz)

    data, targets = data_source[0:len(data_source)-1],data_source[1:]

    out,hidden,decoded, outputs, gates = gated_forward(data, hidden, return_gates=True)
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
        gate_data, hiddens, outputs = evaluate(input_data,batch_size)

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
        gate_data, hiddens, outputs = evaluate(input_data,batch_size)
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


train_percent = 0.30
print('\n=== PLOTTING VIZ ===')
for lyr in range(2):
    """ FORGET GATE """
    # correct
    running_mean_accuracies_correct = [0]*number_of_words

    train_number = int(train_percent*len(forget_gates_correct[lyr]))
    training_indices = np.random.choice(range(len(forget_gates_correct[lyr])),train_number,replace=False)

    fgates = np.array(forget_gates_correct[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_correct[lyr])
    labels = labels[training_indices]

    dc_correct = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(forget_gates_correct[lyr]).reshape(len(forget_gates_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_correct.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)
    ## TODO RANDOMLY SAMPLE FROM FULL DATA TO TRAIN
    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    train_number = int(train_percent*len(forget_gates_incorrect[lyr]))
    training_indices = np.random.choice(range(len(forget_gates_incorrect[lyr])),train_number,replace=False)

    fgates = np.array(forget_gates_incorrect[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_incorrect[lyr])
    labels = labels[training_indices]

    dc_incorrect = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(forget_gates_incorrect[lyr]).reshape(len(forget_gates_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])
    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_incorrect.predict(test_x).item()) == int(labels[x_idx].item()))

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


    """ INPUT GATE """
    # correct
    running_mean_accuracies_correct = [0]*number_of_words

    train_number = int(train_percent*len(input_gates_correct[lyr]))
    training_indices = np.random.choice(range(len(input_gates_correct[lyr])),train_number,replace=False)

    fgates = np.array(input_gates_correct[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_correct[lyr])
    labels = labels[training_indices]

    dc_correct = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(input_gates_correct[lyr]).reshape(len(input_gates_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_correct.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    train_number = int(train_percent*len(input_gates_incorrect[lyr]))
    training_indices = np.random.choice(range(len(input_gates_incorrect[lyr])),train_number,replace=False)

    fgates = np.array(input_gates_incorrect[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_incorrect[lyr])
    labels = labels[training_indices]

    dc_incorrect = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(input_gates_incorrect[lyr]).reshape(len(input_gates_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])
    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_incorrect.predict(test_x).item()) == int(labels[x_idx].item()))

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

    """ OUTPUT GATE """
    # correct
    running_mean_accuracies_correct = [0]*number_of_words

    train_number = int(train_percent*len(output_gates_correct[lyr]))
    training_indices = np.random.choice(range(len(output_gates_correct[lyr])),train_number,replace=False)

    fgates = np.array(output_gates_correct[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_correct[lyr])
    labels = labels[training_indices]

    dc_correct = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(output_gates_correct[lyr]).reshape(len(output_gates_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_correct.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    train_number = int(train_percent*len(output_gates_incorrect[lyr]))
    training_indices = np.random.choice(range(len(output_gates_incorrect[lyr])),train_number,replace=False)

    fgates = np.array(output_gates_incorrect[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_incorrect[lyr])
    labels = labels[training_indices]

    dc_incorrect = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(output_gates_incorrect[lyr]).reshape(len(output_gates_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])
    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_incorrect.predict(test_x).item()) == int(labels[x_idx].item()))

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

    """ CELL GATE """
    # correct
    running_mean_accuracies_correct = [0]*number_of_words

    train_number = int(train_percent*len(cell_states_correct[lyr]))
    training_indices = np.random.choice(range(len(cell_states_correct[lyr])),train_number,replace=False)

    fgates = np.array(cell_states_correct[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_correct[lyr])
    labels = labels[training_indices]

    dc_correct = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(cell_states_correct[lyr]).reshape(len(cell_states_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_correct.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    train_number = int(train_percent*len(cell_states_incorrect[lyr]))
    training_indices = np.random.choice(range(len(cell_states_incorrect[lyr])),train_number,replace=False)

    fgates = np.array(cell_states_incorrect[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_incorrect[lyr])
    labels = labels[training_indices]

    dc_incorrect = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(cell_states_incorrect[lyr]).reshape(len(cell_states_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])
    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_incorrect.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Cell States Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_cell_states.png')
    plt.close()

    """ HIDDEN STATE """
    # correct
    running_mean_accuracies_correct = [0]*number_of_words

    train_number = int(train_percent*len(hidden_states_correct[lyr]))
    training_indices = np.random.choice(range(len(hidden_states_correct[lyr])),train_number,replace=False)

    fgates = np.array(hidden_states_correct[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_correct[lyr])
    labels = labels[training_indices]

    dc_correct = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(hidden_states_correct[lyr]).reshape(len(hidden_states_correct[lyr]),-1)
    labels = np.array(plurality_labels_correct[lyr])

    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_correct.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_correct[idx] = (ct*running_mean_accuracies_correct[idx] + accurate)/(ct+1)

    # incorrect
    running_mean_accuracies_incorrect = [0]*number_of_words

    train_number = int(train_percent*len(hidden_states_incorrect[lyr]))
    training_indices = np.random.choice(range(len(hidden_states_incorrect[lyr])),train_number,replace=False)

    fgates = np.array(hidden_states_incorrect[lyr])
    fgates = fgates[training_indices]
    fgates = fgates.reshape(train_number,-1)

    labels = np.array(plurality_labels_incorrect[lyr])
    labels = labels[training_indices]

    dc_incorrect = sk.LogisticRegression(solver='lbfgs', max_iter=1000).fit(fgates,labels)

    fgates = np.array(hidden_states_incorrect[lyr]).reshape(len(hidden_states_incorrect[lyr]),-1)
    labels = np.array(plurality_labels_incorrect[lyr])
    for x_idx in range(len(fgates)):
        idx = x_idx % number_of_words
        ct = x_idx // number_of_words

        test_x = fgates[x_idx].reshape(1,-1)
        accurate = int(int(dc_incorrect.predict(test_x).item()) == int(labels[x_idx].item()))

        running_mean_accuracies_incorrect[idx] = (ct*running_mean_accuracies_incorrect[idx] + accurate)/(ct+1)

    plt.plot(range(number_of_words),running_mean_accuracies_correct,color='#27ae60')
    plt.scatter(range(number_of_words),running_mean_accuracies_correct,color='#27ae60',label='Correctly Predicted')

    plt.plot(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9')
    plt.scatter(range(number_of_words),running_mean_accuracies_incorrect,color='#2980b9',label='Incorrectly Predicted')
    plt.title('Hidden States Accuracies Over the Sentence (Layer  '+str(lyr)+')')
    plt.xlabel('Position in the Sentence')
    plt.ylabel('Mean Accuracy')
    plt.ylim((-0.05,1.05))
    plt.xticks(range(10))
    plt.legend()
    plt.savefig(save_location+'/lyr_'+str(lyr)+'_hidden_states.png')
    plt.close()
# END OF FILE
