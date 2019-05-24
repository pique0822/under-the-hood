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

ambiguous_cols_full = ['Start','Noun','Ambiguous verb', 'RC contents','Disambiguator']
whowas_ambiguous_cols_full = ['Start','Noun','Unreduced content','Ambiguous verb', 'RC contents','Disambiguator']
unambiguous_cols_full = ['Start','Noun','Unambiguous verb', 'RC contents','Disambiguator']
whowas_unambiguous_cols_full = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents','Disambiguator']



amb_cells = []
ambiguous_targets = []


unamb_cells = []
unambiguous_targets = []


unreamb_cells = []
unreduced_ambiguous_targets = []

unreunamb_cells = []
unreduced_unambiguous_targets = []

amb_temporal_cell_states = {}
unamb_temporal_cell_states = {}

whowas_amb_temporal_cell_states = {}
whowas_unamb_temporal_cell_states = {}

if not load_files:
    print('=== TESTING MODEL ===')
    for df_idx in tqdm(list(range(len(df)))):
        print('')
        row = df.iloc[df_idx]

        main_verb = row['Disambiguator'].strip().lstrip()
        if main_verb in corpus.dictionary.word2idx:
            main_verb_idx = corpus.dictionary.word2idx[main_verb]
        else:
            main_verb_idx = corpus.dictionary.word2idx["<unk>"]

        full_stop_idx = corpus.dictionary.word2idx['.']


        ambiguous_full = ""
        for column in ambiguous_cols_full[:len(ambiguous_cols_full)-1]:
            ambiguous_full += ' '+row[column]
            ambiguous_full = ambiguous_full.lstrip().strip()

        whowas_ambiguous_full = ""
        for column in whowas_ambiguous_cols_full[:len(whowas_ambiguous_cols_full)-1]:
            whowas_ambiguous_full += ' '+row[column]
            whowas_ambiguous_full = whowas_ambiguous_full.lstrip().strip()

        unambiguous_full = ""
        for column in unambiguous_cols_full[:len(unambiguous_cols_full)-1]:
            unambiguous_full += ' '+row[column]
            unambiguous_full = unambiguous_full.lstrip().strip()

        whowas_unambiguous_full = ""
        for column in whowas_unambiguous_cols_full[:len(whowas_unambiguous_cols_full)-1]:
            whowas_unambiguous_full += ' '+row[column]
            whowas_unambiguous_full = whowas_unambiguous_full.lstrip().strip()


        # Ambiguous

        hidden = model.init_hidden(1)

        for column in ambiguous_cols_full:
            partial_input = row[column]
            partial_input = partial_input.lstrip().strip()

            tokenized_inp = corpus.safe_tokenize_sentence(partial_input)

            for token in tokenized_inp:
                input = torch.randint(ntokens, (1, 1), dtype=torch.long)
                input.fill_(token.item())
                output, hidden = model(input,hidden)

            # get the last cell state in the whole temporal region
            if column in amb_temporal_cell_states:
                amb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())
            else:
                amb_temporal_cell_states[column] = []
                amb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())

        ambiguous_surprisal = prefix_to_avg[ambiguous_full]
        ambiguous_targets.append(ambiguous_surprisal)


        # Unambiguous

        hidden = model.init_hidden(1)

        for column in unambiguous_cols_full:
            partial_input = row[column]
            partial_input = partial_input.lstrip().strip()

            tokenized_inp = corpus.safe_tokenize_sentence(partial_input)

            for token in tokenized_inp:
                input = torch.randint(ntokens, (1, 1), dtype=torch.long)
                input.fill_(token.item())
                output, hidden = model(input,hidden)

            # get the last cell state in the whole temporal region
            if column in unamb_temporal_cell_states:
                unamb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())
            else:
                unamb_temporal_cell_states[column] = []
                unamb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())

        unambiguous_surprisal = prefix_to_avg[unambiguous_full
        unambiguous_targets.append(unambiguous_surprisal)


        # Unreduced Ambiguous

        hidden = model.init_hidden(1)

        for column in whowas_ambiguous_cols_full:
            partial_input = row[column]
            partial_input = partial_input.lstrip().strip()

            tokenized_inp = corpus.safe_tokenize_sentence(partial_input)

            for token in tokenized_inp:
                input = torch.randint(ntokens, (1, 1), dtype=torch.long)
                input.fill_(token.item())
                output, hidden = model(input,hidden)

            # get the last cell state in the whole temporal region
            if column in whowas_temporal_cell_states:
                whowas_amb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())
            else:
                whowas_amb_temporal_cell_states[column] = []
                whowas_amb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())

        ambiguous_surprisal = prefix_to_avg[ambiguous_full]
        ambiguous_targets.append(ambiguous_surprisal)


        # Unreduced Unambiguous

        hidden = model.init_hidden(1)

        for column in whowas_unambiguous_cols_full:
            partial_input = row[column]
            partial_input = partial_input.lstrip().strip()

            tokenized_inp = corpus.safe_tokenize_sentence(partial_input)

            for token in tokenized_inp:
                input = torch.randint(ntokens, (1, 1), dtype=torch.long)
                input.fill_(token.item())
                output, hidden = model(input,hidden)

            # get the last cell state in the whole temporal region
            if column in whowas_unamb_temporal_cell_states:
                whowas_unamb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())
            else:
                whowas_unamb_temporal_cell_states[column] = []
                whowas_unamb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())

        whowas_unambiguous_surprisal = prefix_to_avg[whowas_unambiguous_full
        whowas_unambiguous_targets.append(whowas_unambiguous_surprisal)

# generating matrix

generalization_matrix = np.zeros((len(ambiguous_cols_full),len(ambiguous_cols_full)))


training_percent = 0.8
for col,training_column in enumerate(ambiguous_cols_full):

    training_cells = amb_temporal_cell_states[training_column]

    reg = sk.Ridge(alpha=10).fit(training_cells,ambiguous_targets)

    for row,test_column in enumerate(ambiguous_cols_full):
        test_cells = amb_temporal_cell_states[test_column]

        r2 =skm.r2_score(test_cells,ambiguous_targets)

        generalization_matrix[row][col] = max(0,r2)

ax = sns.heatmap(gen_matrix,vmin=0,vmax=1,annot=True, fmt="f")

ax.invert_yaxis()

plt.xticks(np.arange(len(ambiguous_cols_full))+0.5,ambiguous_cols_full)
plt.xlabel('Training On')

plt.yticks(np.arange(len(ambiguous_cols_full))+0.5,ambiguous_cols_full)
plt.ylabel('Testing On')

plt.title('Clipped R^2 Scores')

plt.savefig('ambiguous_generalization_matrix.png')













#EOF
