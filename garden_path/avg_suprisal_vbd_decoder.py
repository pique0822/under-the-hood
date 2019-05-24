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

import argparse

parser = argparse.ArgumentParser(description='Average Surprisal Decoder')
parser.add_argument('--model_path', type=str,
            help='model location', default='/om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt')
parser.add_argument('--seed', type=int, default=1111,
            help='random seed')
parser.add_argument('--training_cells', type=str, default='reduced',
                    help='select the cells that will be used to train the model {reduced | all}')
parser.add_argument('--cross_validation',type=int,default=10,help='Amount of cross validation folds')
parser.add_argument('--file_identifier', type=str, default='TEST',
                    help='unique identifier for the files generated in this method [DO NOT USE SPACES]')
parser.add_argument('--smoothed_significance', type=bool, default=False,
            help='Smoothed significance flag')
args = parser.parse_args()

ROOT = Path(__file__).absolute().parent.parent


load_files = False

if not os.path.exists('best_coefs'):
    os.makedirs('best_coefs')

np.random.seed(args.seed)



# Load the pretrained model
with open(args.model_path, 'rb') as f:
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

    # print(ambiguous_full)
    # break
unamb_cells = np.array(unamb_cells).reshape(len(unamb_cells),-1)
amb_cells = np.array(amb_cells).reshape(len(amb_cells),-1)
unreamb_cells = np.array(unreamb_cells).reshape(len(unreamb_cells),-1)
unreunamb_cells = np.array(unreunamb_cells).reshape(len(unreunamb_cells),-1)

if args.training_cells == 'reduced':
    all_cells = np.concatenate((amb_cells,unamb_cells))

    all_targets = np.concatenate((ambiguous_targets,unambiguous_targets))
elif args.training_cells =='all':
    all_cells = np.concatenate((amb_cells,unamb_cells))
    all_cells = np.concatenate((all_cells,unreamb_cells))
    all_cells = np.concatenate((all_cells,unreunamb_cells))

    all_targets = np.concatenate((ambiguous_targets,unambiguous_targets))
    all_targets = np.concatenate((all_targets,unreduced_ambiguous_targets))
    all_targets = np.concatenate((all_targets,unreduced_unambiguous_targets))


coef_count = {}

shuffled_indices = np.arange(len(all_cells))
np.random.shuffle(shuffled_indices)

print('=== '+str(args.cross_validation)+' Experiment Significant ===')
print('Training on '+str(int((args.cross_validation - 1)/args.cross_validation*len(all_cells))) + ' cell states of '+str(len(all_cells)))

num_runs = 0
all_coefs = None

best_mse_alpha = 0
best_r2_alpha = 0

min_MSE = np.infty
max_MSE = -np.infty

best_mse_reg = None

min_R2 = 1
max_R2 = -np.infty

best_R2_reg = None

for alpha_value in [0.01,0.1,0.2,0.5,1,5,10]:
    print('\nALPHA',alpha_value)

    mean_MSE = 0
    mean_R2 = 0

    for exper_idx in tqdm(list(range(args.cross_validation)), desc="Experiment"):
        # import pdb; pdb.set_trace()
        training_indices = np.concatenate((shuffled_indices[0:int((exper_idx/args.cross_validation)*len(all_cells))],shuffled_indices[int(((exper_idx+1)/args.cross_validation)*len(all_cells)):]))

        test_indices = shuffled_indices[int((exper_idx/args.cross_validation)*len(all_cells)):int(((exper_idx+1)/args.cross_validation)*len(all_cells))]

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
            best_mse_alpha = alpha_value

        if r2 > max_R2:
            max_R2 = r2
            best_R2_reg = reg
            best_r2_alpha = alpha_value

        if r2 < min_R2:
            min_R2 = r2

    print('MEAN HELD OUT R^2',mean_R2)
    print('MEAN HELD OUT MSE',mean_MSE)

print('MAX HELD OUT R^2',max_R2,'alpha',best_r2_alpha)
print('MIN HELD OUT R^2',min_R2)



print('MAX HELD OUT MSE',max_MSE)
print('MIN HELD OUT MSE',min_MSE,'alpha',best_mse_alpha)


if args.smoothed_significance:
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
        # print('True Significant Units',significant_coef_indices)
        # print('Coefficient Signs')
        # predict_ambiguous_sign = []
        #
        # for c in significant_coef_indices:
        #     print(c,reg.coef_[c])
        #     predict_ambiguous_sign.append(np.sign(reg.coef_[c]))

    print('\n\nSMOOTHED BEST R^2 SIGNIFICANCE')
    for c in significant_coef_indices:
        print(c,best_R2_reg.coef_[c])

    print('\n\nSMOOTHED BEST MSE REG SIGNIFICANCE')
    for c in significant_coef_indices:
        print(c,best_mse_reg.coef_[c])

# import pdb; pdb.set_trace()
# mean_coefficients = all_coefs.mean(0)
# std_coefficients = all_coeffs.mean(0)

# REMOVE THIS FOR ORIGINAL
else:
    mean_coef = best_R2_reg.coef_.mean()
    std_coef = best_R2_reg.coef_.std()
    significant_coef_indices = np.where(np.abs(best_R2_reg.coef_) > mean_coef + 3*std_coef)[0]
    print('True Significant Units',significant_coef_indices)
    for c in significant_coef_indices:
        print(c,best_R2_reg.coef_[c])
#######


pickle.dump(best_R2_reg,open('best_coefs/best_r2_coefs_'+args.file_identifier+'.pkl','wb+'))
pickle.dump(best_mse_reg,open('best_coefs/best_mse_coefs_'+args.file_identifier+'.pkl','wb+'))
pickle.dump(significant_coef_indices,open('best_coefs/significant_coefs_'+args.file_identifier+'.pkl','wb+'))

print('Ambiguous R^2 Score',best_R2_reg.score(amb_cells, ambiguous_targets))
print('Unambiguous R^2 Score',best_R2_reg.score(unamb_cells, unambiguous_targets))
print('Unreduced Ambiguous R^2 Score',best_R2_reg.score(unreamb_cells, unreduced_ambiguous_targets))
print('Unreduced Unambiguous R^2 Score',best_R2_reg.score(unreunamb_cells, unreduced_unambiguous_targets))
