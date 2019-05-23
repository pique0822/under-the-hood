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
from pathlib import Path
import utils

import sklearn.metrics as skm

import pickle
from tqdm import tqdm

ROOT = Path(__file__).absolute().parent.parent
model_path = '/om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
save_location = 'DCs'
class_weights = 'balanced'

load_files = False

if not os.path.exists(save_location):
    os.makedirs(save_location)

np.random.seed(1111)



# Load the pretrained model
with open(model_path, 'rb') as f:
    model = torch.load(f, map_location='cpu')

print('=== MODEL INFORMATION ===')
print(model)


data_path = ROOT / "data" / "colorlessgreenRNNs"


print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(data_path)
ntokens = corpus.dictionary.__len__()


df = pd.read_csv(ROOT / "garden_path" / "data" / "verb-ambiguity-with-intervening-phrase.csv")

surp_df = pd.read_csv(ROOT / "evaluate_model" / "surprisal_rc.csv")

total_surprisal = 0
previous_prefix = None
prefix_to_avg = {}
count = 1

for i in range(len(surp_df)):
    split_sentence = surp_df.iloc[i]['sentence'].split(' ')
    surprisal = float(surp_df.iloc[i]['surprisal'])

    prefix = ' '.join(split_sentence[:len(split_sentence)-2])

    if previous_prefix != prefix:
        if previous_prefix is not None:
            prefix_to_avg[previous_prefix] = total_surprisal/count

        total_surprisal = surprisal
        count = 1
    else:
        total_surprisal += surprisal
        count += 1

    previous_prefix = prefix

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

unreunamb_cells = []
unreduced_unambiguous_targets = []

def surprisal_score(verb_surp,period_surp):
    return verb_surp-period_surp

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


        ambiguous_surprisal = prefix_to_avg[ambiguous_full]
        ambiguous_targets.append(ambiguous_surprisal)




        hidden = model.init_hidden(1)

        tokenized_unamb = corpus.safe_tokenize_sentence(unambiguous_prefix.strip())

        for token in tokenized_unamb:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        unamb_cells.append(last_cell)

        unambiguous_surprisal = prefix_to_avg[unambiguous_full]
        unambiguous_targets.append(unambiguous_surprisal)



        hidden = model.init_hidden(1)

        tokenized_unreamb = corpus.safe_tokenize_sentence(whowas_ambiguous_prefix.strip())

        for token in tokenized_unreamb:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)
        last_cell = hidden[1][1].detach().numpy()

        unreamb_cells.append(last_cell)


        whowas_ambiguous_surprisal = prefix_to_avg[whowas_ambiguous_full]
        unreduced_ambiguous_targets.append(whowas_ambiguous_surprisal)



        hidden = model.init_hidden(1)

        tokenized_unreunamb = corpus.safe_tokenize_sentence(whowas_unambiguous_prefix.strip())

        for token in tokenized_unreunamb:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)
        last_cell = hidden[1][1].detach().numpy()

        if whowas_unambiguous_full in prefix_to_avg:
            unreunamb_cells.append(last_cell)


            whowas_unambiguous_surprisal = prefix_to_avg[whowas_unambiguous_full]
            unreduced_unambiguous_targets.append(whowas_unambiguous_surprisal)

        print(ambiguous_full)

unamb_cells = np.array(unamb_cells).reshape(len(unamb_cells),-1)
amb_cells = np.array(amb_cells).reshape(len(amb_cells),-1)
unreamb_cells = np.array(unreamb_cells).reshape(len(unreamb_cells),-1)
unreunamb_cells = np.array(unreunamb_cells).reshape(len(unreunamb_cells),-1)

reduced_cells = np.concatenate((amb_cells,unamb_cells))
reduced_targets = np.concatenate((ambiguous_targets,unambiguous_targets))

all_cells = np.concatenate((amb_cells,unamb_cells))
all_cells = np.concatenate((all_cells,unreamb_cells))
all_cells = np.concatenate((all_cells,unreunamb_cells))

all_targets = np.concatenate((ambiguous_targets,unambiguous_targets))
all_targets = np.concatenate((all_targets,unreduced_ambiguous_targets))
all_targets = np.concatenate((all_targets,unreduced_unambiguous_targets))


