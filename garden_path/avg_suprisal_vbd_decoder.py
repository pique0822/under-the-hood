import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import sklearn.metrics as skm
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import model
import data
import utils

from experiment_util import Experiment, read_surprisal_df


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Average Surprisal Decoder')
parser.add_argument("experiment_file", type=Path, help="path to stimuli csv")
parser.add_argument("surprisal_file", type=Path, help="path to surprisals output for these stimuli")
parser.add_argument('--model_path', type=str,
            help='model location',
            default='/om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt')
parser.add_argument("--data_path", help="path to model data")
parser.add_argument('--seed', type=int, default=1111,
            help='random seed')
parser.add_argument('--smoothed_significance', type=bool, default=False,
            help='Smoothed significance flag')
args = parser.parse_args()


np.random.seed(args.seed)


# Load the pretrained model
with open(args.model_path, 'rb') as f:
    model = torch.load(f, map_location='cpu')

print('=== MODEL INFORMATION ===')
print(model)


print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(args.data_path)
ntokens = len(corpus.dictionary)


experiment = Experiment.from_yaml(args.experiment_file)
surp_df = read_surprisal_df(args.surprisal_file)


# collect average surprisal for a given prefix.
total_surprisal = 0
prefix_counts = Counter()
prefix_to_avg = Counter()

for sentence_id, surprisals in surp_df.groupby("sentence_id"):
    # get surprisal at disambiguator index.
    disambiguator_token_idx = len(surprisals) - 2
    surprisal = list(surprisals.surprisal)[disambiguator_token_idx]

    # we'll average this surprisal over the preceding prefix string content
    prefix = " ".join(list(surprisals.token)[:disambiguator_token_idx]).strip()

    prefix_counts[prefix] += 1
    prefix_to_avg[prefix] += surprisal

# run average
prefix_to_avg = {prefix: total / prefix_counts[prefix]
                 for prefix, total in prefix_to_avg.items()}


results_by_condition = defaultdict(list)

print('=== TESTING MODEL ===')
# Reconstruct the sentences from the original stimuli data now.
# For each condition, retrieve the relevant model state + surprisal.
for idx, row in tqdm(list(experiment.stimuli.iterrows())):
    main_verb = row['Disambiguator'].strip().lstrip()
    if main_verb in corpus.dictionary.word2idx:
        main_verb_idx = corpus.dictionary.word2idx[main_verb]
    else:
        main_verb_idx = corpus.dictionary.word2idx["<unk>"]

    for condition, metadata in experiment.conditions.items():
        # prepare sentence prefix string for lookup
        prefix = " ".join(row[col].strip() for col in metadata["prefix_columns"]).strip()

        try:
            surprisal = prefix_to_avg[prefix]
        except KeyError as e:
            logger.warn("Missing surprisal estimate for prefix: %r", prefix)
            continue

        # from what token idx should we extract a cell state ?
        # NB coupled with tokenization method
        extract_column_idx = metadata["prefix_columns"].index(metadata["extract_column"])
        # get the last token of the region of interest
        extract_token_idx = (sum(len(row[col].split(" "))
                                 for col in metadata["prefix_columns"][:extract_column_idx]) - 1)

        # model emb idx lookup; remove the auto-appended <eos> token
        prefix_tokenized = corpus.safe_tokenize_sentence(prefix)[:-1]

        # extract model state
        hidden = model.init_hidden(1)
        for token in prefix_tokenized:
            input_val = torch.randint(ntokens, (1, 1,), dtype=torch.long)
            input_val.fill_(token.item())
            output, hidden = model(input_val, hidden)

        last_cell = hidden[1][1].detach().numpy().flatten()

        results_by_condition[condition].append((idx, last_cell, surprisal))


cells = {condition: np.array([cell for _, cell, _ in items])
         for condition, items in results_by_condition.items()}
targets = {condition: np.array([surprisal for _, _, surprisal in items])
           for condition, items in results_by_condition.items()}

# items from which conditions should be used to train the decoder?
train_conditions = experiment.decoder_config["train_conditions"]

X = np.concatenate([cells[cond] for cond in train_conditions])
y = np.concatenate([targets[cond] for cond in train_conditions])
n = len(X)


coef_count = {}

shuffled_indices = np.arange(len(X))
np.random.shuffle(shuffled_indices)

cv_folds = experiment.decoder_config["cross_validation_folds"]
print('=== '+str(cv_folds)+' Experiment Significant ===')
print('Training on %i cell states of %i' % (int((cv_folds - 1)/cv_folds*n), n))

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

    for exper_idx in tqdm(list(range(cv_folds)), desc="Experiment"):
        # import pdb; pdb.set_trace()
        training_indices = np.concatenate((shuffled_indices[0:int((exper_idx/cv_folds)*n)],shuffled_indices[int(((exper_idx+1)/cv_folds)*n):]))

        test_indices = shuffled_indices[int((exper_idx/cv_folds)*n):int(((exper_idx+1)/cv_folds)*n)]

        num_runs += 1

        reg = sk.Ridge(alpha=alpha_value).fit(X[training_indices], y[training_indices])

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

        predicted_surp = reg.predict(X[test_indices])

        # print('EXP '+str(exper_idx)+':: Held Out R^2 Score',skm.r2_score(all_targets[test_indices],predicted_surp))

        r2 =skm.r2_score(y[test_indices],predicted_surp)

        mean_R2 = (mean_R2*exper_idx + r2)/(exper_idx+1)

        # print('EXP '+str(exper_idx)+':: Held Out MSE Score',skm.mean_squared_error(all_targets[test_indices],predicted_surp))
        mse = skm.mean_squared_error(y[test_indices],predicted_surp)

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

# import pdb; pdb.set_trace()
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
else:
    mean_coef = best_R2_reg.coef_.mean()
    std_coef = best_R2_reg.coef_.std()
    significant_coef_indices = np.where(np.abs(best_R2_reg.coef_) > mean_coef + 3*std_coef)[0]
    print('True Significant Units',significant_coef_indices)
    for c in significant_coef_indices:
        print(c,best_R2_reg.coef_[c])
#######
# mean_coef = best_R2_reg.coef_.mean()
# std_coef = best_R2_reg.coef_.std()
# significant_coef_indices = np.where(np.abs(best_R2_reg.coef_) > mean_coef + 3*std_coef)[0]
# print('True Significant Units',significant_coef_indices)
# for c in significant_coef_indices:
#     print(c,best_R2_reg.coef_[c])

pickle.dump(best_R2_reg,open('best_r2_coefs.pkl','wb+'))
pickle.dump(best_mse_reg,open('best_mse_coefs.pkl','wb+'))
pickle.dump(significant_coef_indices,open('significant_coefs.pkl','wb+'))

for condition in experiment.conditions:
    X_test, y_test = cells[condition], targets[condition]
    print("R^2 score for condition %s: %f" % (condition, best_R2_reg.score(X_test, y_test)))
