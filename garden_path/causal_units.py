import pandas as pd
import utils
import dictionary_corpus
import torch

def causal_get_probability_vector(model,dictionary, sentence,causal_unit=None):
    ntokens = dictionary.__len__()
    prefix = utils.tokenize_sentence(dictionary, sentence)

    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')

    firstword = prefix[0]
    input.fill_(firstword.item())
    output, hidden = gated_forward(input,hidden,causal_unit=causal_unit)
    word_weights = output.squeeze().div(1).exp().cpu()
    probs = word_weights/sum(word_weights)
    # import pdb; pdb.set_trace()
    for word in prefix[1:len(prefix)]:
        prob = probs[word].item()
        input.fill_(word.item())
        output, hidden = gated_forward(input,hidden,causal_unit=causal_unit)
        word_weights = output.squeeze().div(1).exp().cpu()
        probs = word_weights/sum(word_weights)

    return probs

def causal_most_likely_continuation(model, dictionary, prefix, possible_conts, memoize = {}, return_prob = True, causal_unit=None):
    if prefix in memoize:
        probs = memoize[prefix]
    else:
        probs = causal_get_probability_vector(model, dictionary, prefix, causal_unit)
        memoize[prefix] = probs
    tokenized_conts = []
    for continuation in possible_conts:
        tokenized_conts.append(utils.tokenize_sentence(dictionary, continuation))

    cont_probs = []
    largest_prob = 0
    largest_prob_idx = 0
    for idx, tconts in enumerate(tokenized_conts):
        prob = probs[tconts].item()
        cont_probs.append(prob)

        if prob > largest_prob:
            largest_prob = prob
            largest_prob_idx = idx

    cont_probs
    if return_prob == False:
        for i in range(len(cont_probs)):
            p = cont_probs[i]
            s = utils.probability_to_surprisal(p)
            cont_probs[i] = s

    return largest_prob_idx, cont_probs, memoize

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
xl = pd.read_csv('data/verb-ambiguity-with-intervening-phrase.csv')

with open(model_path, 'rb') as f:
    model = torch.load(f, map_location='cpu')

data_path = "../data/colorlessgreenRNNs/"

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
def gated_forward(input, hidden, return_gates=False, causal_unit=None):
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

        causal_eye = torch.eye(h0_l0.shape[1])

        # This way, we will only zero one unit and not the multiple over all layers
        if causal_unit is not None:
            if causal_unit < h0_l0.shape[1]:
                causal_eye[causal_unit][causal_unit] = 0
            else:
                causal_unit -= h0_l0.shape[1]

        f_gates, i_gates, o_gates, g_gates = [],[],[],[]
        for seq_i in range(len(raw_output)):
            inp = raw_output[seq_i]

            # forget gate
            f_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_if_l0)) + b_if_l0) + (torch.matmul(h0_l0,torch.t(w_hf_l0)) + b_hf_l0))
            f_g_l0 = torch.matmul(f_g_l0,causal_eye)

            # input gate
            i_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_ii_l0)) + b_ii_l0) + (torch.matmul(h0_l0,torch.t(w_hi_l0)) + b_hi_l0))

            i_g_l0 = torch.matmul(i_g_l0,causal_eye)

            # output gate
            o_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_io_l0)) + b_io_l0) + (torch.matmul(h0_l0, torch.t(w_ho_l0)) + b_ho_l0))

            o_g_l0 = torch.matmul(o_g_l0,causal_eye)

            # intermediate cell state
            c_tilde_l0 = torch.tanh((torch.matmul(inp,torch.t(w_ig_l0)) + b_ig_l0) + (torch.matmul(h0_l0, torch.t(w_hg_l0)) + b_hg_l0))

            c_tilde_l0 = torch.matmul(c_tilde_l0,causal_eye)

            # current cell state
            c0_l0 = f_g_l0 * c0_l0 + i_g_l0 * c_tilde_l0
            c0_l0 = torch.matmul(c0_l0,causal_eye)

            # hidden state
            h0_l0 = o_g_l0 * torch.tanh(c0_l0)
            h0_l0 = torch.matmul(h0_l0,causal_eye)

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
    output = model.drop(raw_output)
    # import pdb; pdb.set_trace()
    decoded = model.decoder(output.view(1, 650))

    return decoded.view(1, 1, decoded.size(1)), hidden

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

dictionary = dictionary_corpus.Dictionary(data_path)

with open('causal_results.txt','w+') as file:
    memoize = {}
    for i in range(len(xl)):
        file.write('\n\n')
        print('\n\n')
        row = xl.iloc[i]
        base_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Ambiguous verb'].strip() + ' ' + row['RC contents'].strip()

        alternate_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Unambiguous verb'].strip() + ' ' + row['RC contents'].strip()

        main_verb = row['Disambiguator'].strip()

        file.write(row['Ambiguous verb'].strip() + ' vs '+row['Unambiguous verb'].strip()+'\n')
        print(row['Ambiguous verb'].strip() + ' vs '+row['Unambiguous verb'].strip()+'\n')

        largest_prob_idx, cont_surps, memoize = utils.most_likely_continuation(model, dictionary,base_sentence,['.',main_verb],memoize, return_prob = False)
        period_surp, verb_surp = cont_surps
        base_processing_issue = period_surp-verb_surp
        file.write('Baseline ' + str(base_processing_issue)+'\n')
        print('Baseline ' + str(base_processing_issue)+'\n')
        for unit in range(1300):
            largest_prob_idx, cont_surps, gated_mem = causal_most_likely_continuation(model, dictionary,base_sentence,['.',main_verb], return_prob = False, memoize = {}, causal_unit = unit)
            period_surp, verb_surp = cont_surps
            gated_processing_issue = period_surp-verb_surp
            percent = (gated_processing_issue - base_processing_issue)/base_processing_issue
            if percent > 10 or percent < -10:
                file.write('* Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
                print('* Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
            else:
                file.write('Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
                print('Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')

        largest_prob_idx, cont_surps, memoize = utils.most_likely_continuation(model, dictionary,alternate_sentence,['.',main_verb],memoize={}, return_prob = False)
        period_surp, verb_surp = cont_surps
        alt_processing_issue = period_surp-verb_surp
        file.write('Alternate ' + str(alt_processing_issue)+'\n')
        print('Alternate ' + str(alt_processing_issue)+'\n')
        gated_mem = {}
        for unit in range(1300):
            largest_prob_idx, cont_surps, gated_mem = causal_most_likely_continuation(model, dictionary,base_sentence,['.',main_verb],gated_mem, return_prob = False, causal_unit = unit)
            period_surp, verb_surp = cont_surps
            gated_processing_issue = period_surp-verb_surp

            if percent > 10 or percent < -10:
                file.write('* Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
                print('* Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
            else:
                file.write('Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
                print('Unit '+str(unit) + ' : ' + str(round(percent,4))+'%\n')
