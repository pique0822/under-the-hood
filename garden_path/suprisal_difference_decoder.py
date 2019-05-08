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


data_path = "../data/colorlessgreenRNNs/"


print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(data_path)
ntokens = corpus.dictionary.__len__()


df = pd.read_csv('data/verb-ambiguity-with-intervening-phrase.csv')


ambiguous_cols_prefix = ['Start','Noun','Ambiguous verb']
whowas_ambiguous_cols_prefix = ['Start','Noun','Unreduced content','Ambiguous verb']
unambiguous_cols_prefix = ['Start','Noun','Unambiguous verb']
whowas_unambiguous_cols_prefix = ['Start','Noun','Unreduced content','Unambiguous verb']

ambiguous_cols_full = ['Start','Noun','Ambiguous verb', 'RC contents']
whowas_ambiguous_cols_full = ['Start','Noun','Unreduced content','Ambiguous verb', 'RC contents']
unambiguous_cols_full = ['Start','Noun','Unambiguous verb', 'RC contents']
whowas_unambiguous_cols_full = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents']



amb_cells = []
ambiguous_targets = []


unamb_cells = []
unambiguous_targets = []


unreamb_cells = []
unreduced_ambiguous_targets = []

def scoring_function(y_pred,y_true,round=5):
    # this is the absolute error difference
    return np.round_((y_pred - y_true)/y_true,round)

if not load_files:
    print('=== TESTING MODEL ===')
    for df_idx in range(len(df)):
        print('')
        row = df.iloc[df_idx]

        main_verb = row['Disambiguator'].strip().lstrip()
        if main_verb in corpus.dictionary.word2idx:
            main_verb_idx = corpus.dictionary.word2idx[main_verb]
        else:
            main_verb_idx = corpus.dictionary.word2idx["<unk>"]

        full_stop_idx = corpus.dictionary.word2idx['.']

        ambiguous_prefix = ""
        for column in ambiguous_cols_prefix:
            ambiguous_prefix += ' '+row[column]
            ambiguous_prefix = ambiguous_prefix.lstrip().strip()

        whowas_ambiguous_prefix = ""
        for column in whowas_ambiguous_cols_prefix:
            whowas_ambiguous_prefix += ' '+row[column]
            whowas_ambiguous_prefix = whowas_ambiguous_prefix.lstrip().strip()

        unambiguous_prefix = ""
        for column in unambiguous_cols_prefix:
            unambiguous_prefix += ' '+row[column]
            unambiguous_prefix = unambiguous_prefix.lstrip().strip()

        whowas_unambiguous_prefix = ""
        for column in whowas_unambiguous_cols_prefix:
            whowas_unambiguous_prefix += ' '+row[column]
            whowas_unambiguous_prefix = whowas_unambiguous_prefix.lstrip().strip()




        ambiguous_full = ""
        for column in ambiguous_cols_full:
            ambiguous_full += ' '+row[column]
            ambiguous_full = ambiguous_full.lstrip().strip()

        whowas_ambiguous_full = ""
        for column in whowas_ambiguous_cols_full:
            whowas_ambiguous_full += ' '+row[column]
            whowas_ambiguous_full = whowas_ambiguous_full.lstrip().strip()

        unambiguous_full = ""
        for column in unambiguous_cols_full:
            unambiguous_full += ' '+row[column]
            unambiguous_full = unambiguous_full.lstrip().strip()

        whowas_unambiguous_full = ""
        for column in whowas_unambiguous_cols_full:
            whowas_unambiguous_full += ' '+row[column]
            whowas_unambiguous_full = whowas_unambiguous_full.lstrip().strip()





        tokenized_amb = corpus.safe_tokenize_sentence(ambiguous_prefix.strip())

        hidden = model.init_hidden(1)

        for token in tokenized_amb:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        amb_cells.append(last_cell)

        # MOST LIKELY CONTINUATION
        hidden = model.init_hidden(1)

        tokenized_amb_full = corpus.safe_tokenize_sentence(ambiguous_full.strip())

        for token in tokenized_amb_full:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        word_weights = output.squeeze().div(1).exp().cpu()
        word_surprisals = -1*torch.log2(word_weights/sum(word_weights))

        verb_surp = word_surprisals[main_verb_idx].item()
        period_surp = word_surprisals[full_stop_idx].item()


        ambiguous_surprisal = verb_surp - period_surp
        ambiguous_targets.append(ambiguous_surprisal)




        hidden = model.init_hidden(1)

        tokenized_unamb = corpus.safe_tokenize_sentence(unambiguous_prefix.strip())

        for token in tokenized_unamb:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        unamb_cells.append(last_cell)

        # MOST LIKELY CONTINUATION
        hidden = model.init_hidden(1)

        tokenized_unamb_full = corpus.safe_tokenize_sentence(unambiguous_full.strip())

        for token in tokenized_unamb_full:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        word_weights = output.squeeze().div(1).exp().cpu()
        word_surprisals = -1*torch.log2(word_weights/sum(word_weights))

        verb_surp = word_surprisals[main_verb_idx].item()
        period_surp = word_surprisals[full_stop_idx].item()

        unambiguous_surprisal = verb_surp - period_surp
        unambiguous_targets.append(unambiguous_surprisal)



        hidden = model.init_hidden(1)

        tokenized_unreamb = corpus.safe_tokenize_sentence(whowas_ambiguous_prefix.strip())

        for token in tokenized_unreamb:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)
        last_cell = hidden[1][1].detach().numpy()

        unreamb_cells.append(last_cell)


        # MOST LIKELY CONTINUATION
        hidden = model.init_hidden(1)

        tokenized_unreamb_full = corpus.safe_tokenize_sentence(whowas_unambiguous_full.strip())

        for token in tokenized_unreamb_full:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        word_weights = output.squeeze().div(1).exp().cpu()
        word_surprisals = -1*torch.log2(word_weights/sum(word_weights))

        verb_surp = word_surprisals[main_verb_idx].item()
        period_surp = word_surprisals[full_stop_idx].item()

        whowas_ambiguous_surprisal = verb_surp - period_surp
        unreduced_ambiguous_targets.append(whowas_ambiguous_surprisal)



        print(ambiguous_full)
        # break
if not load_files:
    unamb_cells = np.array(unamb_cells).reshape(len(unamb_cells),-1)
    amb_cells = np.array(amb_cells).reshape(len(amb_cells),-1)
    unreamb_cells = np.array(unreamb_cells).reshape(len(unreamb_cells),-1)

print('\n=== PLOTTING VIZ ===')
if not load_files:
    np.save('data/saved_arrays/unamb_cell_states.npy', unamb_cells)
    np.save('data/saved_arrays/unamb_surp_diff.npy', unambiguous_targets)

    np.save('data/saved_arrays/amb_cell_states.npy', amb_cells)
    np.save('data/saved_arrays/amb_surp_diff.npy', ambiguous_targets)


    np.save('data/saved_arrays/unreamb_cell_states.npy', unreamb_cells)
    np.save('data/saved_arrays/unreamb_surp_diff.npy', unreduced_ambiguous_targets)
else:
    unamb_cells = np.load('data/saved_arrays/unamb_cell_states.npy')
    unambiguous_targets = np.load('data/saved_arrays/unamb_surp_diff.npy')

    amb_cells = np.load('data/saved_arrays/amb_cell_states.npy')
    ambiguous_targets = np.load('data/saved_arrays/amb_surp_diff.npy')

    unreamb_cells = np.load('data/saved_arrays/unreamb_cell_states.npy')
    unreduced_ambiguous_targets = np.load('data/saved_arrays/unreamb_surp_diff.npy')


all_cells = np.concatenate((amb_cells,unamb_cells))
all_targets = np.concatenate((ambiguous_targets,unambiguous_targets))

# END OF FILE

coef_count = {}
train_percent = .7
num_experiments = 1000


print('=== '+str(num_experiments)+' Experiment Significant ===')
print('Training on '+str(int(train_percent*len(all_cells))) + ' cell states of '+str(len(all_cells)))
num_runs = 0
for exper_idx in range(num_experiments):

    training_indices = np.random.choice(range(len(all_cells)),int(train_percent*len(all_cells)),replace=False)

    num_runs += 1

    reg = sk.LinearRegression().fit(all_cells[training_indices],all_targets[training_indices])


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

# This was added becaues only one coefficient comes up as significant
if len(coef_count) == 1:
    significant_coef_indices = list(coef_count.keys())
print('True Significant Units',significant_coef_indices)
print('Coefficient Signs')
predict_ambiguous_sign = []

for c in significant_coef_indices:
    print(c,np.sign(reg.coef_[c]))
    predict_ambiguous_sign.append(np.sign(reg.coef_[c]))

print('Ambiguous R^2 Score',reg.score(amb_cells, ambiguous_targets))
print('Unambiguous R^2 Score',reg.score(unamb_cells, unambiguous_targets))
print('Unreduced Ambiguous R^2 Score',reg.score(unreamb_cells, unreduced_ambiguous_targets))
