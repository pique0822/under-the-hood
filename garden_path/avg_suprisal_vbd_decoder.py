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
        # break
if not load_files:
    unamb_cells = np.array(unamb_cells).reshape(len(unamb_cells),-1)
    amb_cells = np.array(amb_cells).reshape(len(amb_cells),-1)
    unreamb_cells = np.array(unreamb_cells).reshape(len(unreamb_cells),-1)
    unreunamb_cells = np.array(unreunamb_cells).reshape(len(unreunamb_cells),-1)

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
#
# # uncomment for all sentences in training
# all_cells = np.concatenate((all_cells,unreamb_cells))
# all_cells = np.concatenate((all_cells,unreunamb_cells))

# all cells
# all_cells = np.concatenate((unreamb_cells,unreunamb_cells))

all_targets = np.concatenate((ambiguous_targets,unambiguous_targets))
#
# # uncomment for all sentences in training
# all_targets = np.concatenate((all_targets,unreduced_ambiguous_targets))
# all_targets = np.concatenate((all_targets,unreduced_unambiguous_targets))

# just unreduced
# all_targets = np.concatenate((unreduced_ambiguous_targets,unreduced_unambiguous_targets))

""" Jsut reduced
39 -0.38784295
189 0.33676547
281 -0.47673437 *
474 0.32674462 *
648 -0.34725198
"""
""" all
474 0.28397393 *
396 -0.2829467 &
499 -0.23890975
27 0.29303062
281 -0.33377486 *
"""
""" unreduced
396 -0.3606371 &
533 0.16612066
51 0.18047051
115 -0.23789778
93 -0.18395881
"""
""" 2std
16 0.37091076
326 0.12533382
396 -0.39716858 &
404 0.24766985
533 0.47737217
51 0.3803634
115 -0.3785558
93 -0.5350999
91 -0.2485853
"""
# END OF FILE

coef_count = {}
train_percent = .7
num_experiments = 1000


print('=== '+str(num_experiments)+' Experiment Significant ===')
print('Training on '+str(int(train_percent*len(all_cells))) + ' cell states of '+str(len(all_cells)))
num_runs = 0
all_coefs = None

mean_MSE = 0
mean_R2 = 0

min_MSE = np.infty
max_MSE = -np.infty

best_mse_reg = None

min_R2 = 1
max_R2 = -np.infty

best_R2_reg = None

for alpha_value in [0.01,0.1,0.2,0.5,1,5,10]:
    print('\nALPHA',alpha_value)
    for exper_idx in range(num_experiments):

        training_indices = np.random.choice(range(len(all_cells)),int(train_percent*len(all_cells)),replace=False)

        test_indices = list(set(range(len(all_cells))).difference(training_indices))

        num_runs += 1

        reg = sk.Ridge(alpha=alpha_value).fit(all_cells[training_indices],all_targets[training_indices])

        if all_coefs is None:
            all_coefs = reg.coef_.copy()
        else:
            all_coefs = np.vstack((all_coefs,reg.coef_))

        mean_coef = reg.coef_.mean()
        std_coef = reg.coef_.std()
        significant_coef_indices = np.where(np.abs(reg.coef_) > mean_coef + 3*std_coef)[0]
        for coef_idx in significant_coef_indices:
            if coef_idx in coef_count:
                coef_count[coef_idx] += 1
            else:
                coef_count[coef_idx] = 1

        predicted_surp = reg.predict(all_cells[test_indices])

        # print('EXP '+str(exper_idx)+':: Held Out R^2 Score',skm.r2_score(all_targets[test_indices],predicted_surp))

        r2 =skm.r2_score(all_targets[test_indices],predicted_surp)

        mean_R2 = (mean_R2*exper_idx + r2)/(exper_idx+1)

        # print('EXP '+str(exper_idx)+':: Held Out MSE Score',skm.mean_squared_error(all_targets[test_indices],predicted_surp))
        mse = skm.mean_squared_error(all_targets[test_indices],predicted_surp)

        mean_MSE = (mean_MSE*exper_idx + mse)/(exper_idx + 1)

        if mse > max_MSE:
            max_MSE = mse

        if mse < min_MSE:
            min_MSE = mse
            best_mse_reg = reg

        if r2 > max_R2:
            max_R2 = r2
            best_R2_reg = reg

        if r2 < min_R2:
            min_R2 = r2

    print('MEAN HELD OUT R^2',mean_R2)
    print('MEAN HELD OUT MSE',mean_MSE)

    print('MAX HELD OUT R^2',max_R2)
    print('MIN HELD OUT R^2',min_R2)


    print('MAX HELD OUT MSE',max_MSE)
    print('MIN HELD OUT MSE',min_MSE)

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
        print(c,reg.coef_[c])
        predict_ambiguous_sign.append(np.sign(reg.coef_[c]))

    print('\n\nBEST R^2 REG')
    for c in significant_coef_indices:
        print(c,best_R2_reg.coef_[c])

    print('\n\nBEST MSE REG')
    for c in significant_coef_indices:
        print(c,best_mse_reg.coef_[c])

# import pdb; pdb.set_trace()
# mean_coefficients = all_coefs.mean(0)
# std_coefficients = all_coeffs.mean(0)

# REMOVE THIS FOR ORIGINAL
# mean_coef = best_R2_reg.coef_.mean()
# std_coef = best_R2_reg.coef_.std()
# significant_coef_indices = np.where(np.abs(best_R2_reg.coef_) > mean_coef + 3*std_coef)[0]
#######


pickle.dump(best_R2_reg,open('best_coefs/best_r2_coefs_vbd_FINAL.pkl','wb+'))
pickle.dump(best_mse_reg,open('best_coefs/best_mse_coefs_vbd_FINAL.pkl','wb+'))
pickle.dump(significant_coef_indices,open('best_coefs/significant_coefs_vbd_FINAL.pkl','wb+'))

print('Ambiguous R^2 Score',best_R2_reg.score(amb_cells, ambiguous_targets))
print('Unambiguous R^2 Score',best_R2_reg.score(unamb_cells, unambiguous_targets))
print('Unreduced Ambiguous R^2 Score',best_R2_reg.score(unreamb_cells, unreduced_ambiguous_targets))
print('Unreduced Unambiguous R^2 Score',best_R2_reg.score(unreunamb_cells, unreduced_unambiguous_targets))
