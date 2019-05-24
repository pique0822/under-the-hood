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

import seaborn as sns

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
        if ambiguous_full in prefix_to_avg:
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
        if unambiguous_full in prefix_to_avg:
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


            unambiguous_surprisal = prefix_to_avg[unambiguous_full]
            unambiguous_targets.append(unambiguous_surprisal)


        # Unreduced Ambiguous
        if whowas_unambiguous_full in prefix_to_avg:
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
                if column in whowas_amb_temporal_cell_states:
                    whowas_amb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())
                else:
                    whowas_amb_temporal_cell_states[column] = []
                    whowas_amb_temporal_cell_states[column].append(hidden[1][1].detach().numpy())


            unreduced_ambiguous_surprisal = prefix_to_avg[whowas_unambiguous_full]
            unreduced_ambiguous_targets.append(unreduced_ambiguous_surprisal)


        # Unreduced Unambiguous
        if whowas_unambiguous_full in prefix_to_avg:
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

            unreduced_unambiguous_surprisal = prefix_to_avg[whowas_unambiguous_full]
            unreduced_unambiguous_targets.append(unreduced_unambiguous_surprisal)
# generating matrix


def generalization_matrix(relevant_cols,temporal_cell_states,targets,save_name,training_percent = 0.8):

    generalization_matrix = np.zeros((len(relevant_cols),len(relevant_cols)))



    for col,training_column in enumerate(relevant_cols):

        training_cells =temporal_cell_states[training_column]

        training_cells = np.array(training_cells).reshape(len(training_cells),-1)

        reg = sk.Ridge(alpha=10).fit(training_cells,targets)

        for row,test_column in enumerate(relevant_cols):
            test_cells = temporal_cell_states[test_column]

            test_cells = np.array(test_cells).reshape(len(test_cells),-1)

            predicted_surp = reg.predict(test_cells)
            r2 =skm.r2_score(targets,predicted_surp)

            generalization_matrix[row][col] = max(0,r2)

    ax = sns.heatmap(generalization_matrix,vmin=0,vmax=1,annot=True, fmt="f")

    ax.invert_yaxis()
    ax.tick_params( labelsize='small', labelrotation=45)
    plt.xticks(np.arange(len(relevant_cols))+0.5,relevant_cols)
    plt.xlabel('Training On')

    plt.yticks(np.arange(len(relevant_cols))+0.5,relevant_cols)
    plt.ylabel('Testing On')

    plt.title('Clipped R^2 Scores')
    plt.tight_layout()

    plt.savefig(save_name+'.png')

    plt.close()



generalization_matrix(ambiguous_cols_full,amb_temporal_cell_states,ambiguous_targets,'generalization_matrix_ambiguous_reduced')

generalization_matrix(unambiguous_cols_full,unamb_temporal_cell_states,unambiguous_targets,'generalization_matrix_unambiguous_reduced')

generalization_matrix(whowas_ambiguous_cols_full,whowas_amb_temporal_cell_states,unreduced_ambiguous_targets,'generalization_matrix_ambiguous_unreduced')

generalization_matrix(whowas_unambiguous_cols_full,whowas_unamb_temporal_cell_states,unreduced_unambiguous_targets,'generalization_matrix_unambiguous_unreduced')
