import torch
from torch import nn
from torch.nn import functional as F

import model
import data

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import pandas as pd

import os
import utils

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
save_location = '2D_viz'
class_weights = 'balanced'

load_files = False

if not os.path.exists(save_location):
    os.makedirs(save_location)

# np.random.seed(42)

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


ambiguous_cols = ['Start','Noun','Ambiguous verb']
whowas_ambiguous_cols = ['Start','Noun','Unreduced content','Ambiguous verb']
unambiguous_cols = ['Start','Noun','Unambiguous verb']
whowas_unambiguous_cols = ['Start','Noun','Unreduced content','Unambiguous verb']

ambiguous_cells = []
whowas_ambiguous_cells = []
unambiguous_cells = []
whowas_unambiguous_cells = []

all_cells = []
# will store ambiguous vs not ambiguous
all_labels = []


if not load_files:
    print('\n=== TESTING MODEL ===')
    for df_idx in range(len(df)):
        print('')
        row = df.iloc[df_idx]

        hidden = model.init_hidden(1)

        ambiguous_sentence = ""
        for column in ambiguous_cols:
            ambiguous_sentence += ' '+row[column]
            ambiguous_sentence = ambiguous_sentence.lstrip().strip()

        whowas_ambiguous_sentence = ""
        for column in whowas_ambiguous_cols:
            whowas_ambiguous_sentence += ' '+row[column]
            whowas_ambiguous_sentence = whowas_ambiguous_sentence.lstrip().strip()

        unambiguous_sentence = ""
        for column in unambiguous_cols:
            unambiguous_sentence += ' '+row[column]
            unambiguous_sentence = unambiguous_sentence.lstrip().strip()

        whowas_unambiguous_sentence = ""
        for column in whowas_unambiguous_cols:
            whowas_unambiguous_sentence += ' '+row[column]
            whowas_unambiguous_sentence = whowas_unambiguous_sentence.lstrip().strip()

        print(ambiguous_sentence)

        # tokenizing sentences
        tokenized_ambiguous = corpus.safe_tokenize_sentence(ambiguous_sentence.strip())
        tokenized_whowas_ambiguous = corpus.safe_tokenize_sentence(whowas_ambiguous_sentence.strip())

        tokenized_unambiguous = corpus.safe_tokenize_sentence(unambiguous_sentence.strip())
        tokenized_whowas_unambiguous = corpus.safe_tokenize_sentence(whowas_unambiguous_sentence.strip())



        # getting results
        hidden = model.init_hidden(1)

        for token in tokenized_ambiguous:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        ambiguous_cells.append(last_cell)



        hidden = model.init_hidden(1)

        for token in tokenized_whowas_ambiguous:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        # all_cells.append(last_cell)
        whowas_ambiguous_cells.append(last_cell)
        # all_labels.append(0)



        hidden = model.init_hidden(1)

        for token in tokenized_unambiguous:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        # all_cells.append(last_cell)
        unambiguous_cells.append(last_cell)
        # all_labels.append(0)



        hidden = model.init_hidden(1)

        for token in tokenized_whowas_unambiguous:
            input = torch.randint(ntokens, (1, 1), dtype=torch.long)
            input.fill_(token.item())
            output, hidden = model(input,hidden)

        last_cell = hidden[1][1].detach().numpy()

        # all_cells.append(last_cell)
        whowas_unambiguous_cells.append(last_cell)
        # all_labels.append(0)

print('\n=== EVALUATION ===')
ambiguous_cells = np.array(ambiguous_cells)
ambiguous_cells = ambiguous_cells.reshape(len(ambiguous_cells),-1)

whowas_ambiguous_cells = np.array(whowas_ambiguous_cells)
whowas_ambiguous_cells = whowas_ambiguous_cells.reshape(len(whowas_ambiguous_cells),-1)

unambiguous_cells = np.array(unambiguous_cells)
unambiguous_cells = unambiguous_cells.reshape(len(unambiguous_cells),-1)

whowas_unambiguous_cells = np.array(whowas_unambiguous_cells)
whowas_unambiguous_cells = whowas_unambiguous_cells.reshape(len(whowas_unambiguous_cells),-1)


all_cells = ambiguous_cells.copy()

all_cells =  np.vstack((all_cells,unambiguous_cells))

all_labels = np.array( [1]*len(ambiguous_cells)+[0]*(len(all_cells) - len(ambiguous_cells)))


coef_count = {}
train_percent = .7
num_experiments = 1000


print('=== '+str(num_experiments)+' Experiment Significant ===')
print('Training on '+str(int(train_percent*len(all_cells))) + ' cell states of '+str(len(all_cells)))
num_runs = 0
for exper_idx in range(num_experiments):


    positive_train_number = int(train_percent*len(ambiguous_cells))

    positive_training_indices = np.random.choice(range(len(ambiguous_cells)),positive_train_number,replace=False)

    negative_train_number = int(train_percent*(len(all_cells)-len(ambiguous_cells)))

    negative_training_indices = np.random.choice(range(len(all_cells)-len(ambiguous_cells)),negative_train_number,replace=False)

    negative_training_indices = negative_training_indices+len(ambiguous_cells)

    training_indices = np.concatenate((positive_training_indices,negative_training_indices))


    if not sum(all_labels[training_indices]) >= 1:
        continue

    num_runs += 1

    reg = sk.LogisticRegression(solver='lbfgs',class_weight='balanced').fit(all_cells[training_indices],all_labels[training_indices])

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
print('True Significant Units',significant_coef_indices)

# The only consistent coeficients [100, 499, 277]
# [-,-,+]

print('Coefficient Signs')
predict_ambiguous_sign = []

for c in significant_coef_indices:
    print(c,np.sign(reg.coef_[0,c]))
    predict_ambiguous_sign.append(np.sign(reg.coef_[0,c]))

# predicting using regression between ambiguous and unambiguous

print('\n')
label = np.ones(len(ambiguous_cells))
print('Ambiguous Sentence Prediction Accuracy',reg.score(ambiguous_cells,label))

label = np.zeros(len(whowas_ambiguous_cells))
print('Who Was Ambiguous Sentence Prediction Accuracy',reg.score(whowas_ambiguous_cells,label))

label = np.zeros(len(unambiguous_cells))
print('Unambiguous Sentence Prediction Accuracy',reg.score(unambiguous_cells,label))

label = np.zeros(len(whowas_unambiguous_cells))
print('Who Was Unambiguous Sentence Prediction Accuracy',reg.score(whowas_unambiguous_cells,label))

# MODIFYING CELL STATES

modified_ambiguous_cells = ambiguous_cells.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    for row in range(modified_ambiguous_cells.shape[0]):
        modified_ambiguous_cells[row][coef_idx] = -predict_ambiguous_sign[idx]

modified_unambiguous_cells = unambiguous_cells.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    for row in range(modified_unambiguous_cells.shape[0]):
        modified_unambiguous_cells[row][coef_idx] = predict_ambiguous_sign[idx]


print('\n\n REDUCED ACCURACY \n\n')
label = np.ones(len(modified_ambiguous_cells))
print('MOD Ambiguous Sentence Prediction Accuracy',reg.score(modified_ambiguous_cells,label))

label = np.zeros(len(modified_unambiguous_cells))
print('MOD Unambiguous Sentence Prediction Accuracy',reg.score(modified_unambiguous_cells,label))




modified_ambiguous_cells = ambiguous_cells.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    for row in range(modified_ambiguous_cells.shape[0]):
        modified_ambiguous_cells[row][coef_idx] = predict_ambiguous_sign[idx]

modified_unambiguous_cells = unambiguous_cells.copy()
for idx in range(len(significant_coef_indices)):
    coef_idx = significant_coef_indices[idx]
    for row in range(modified_unambiguous_cells.shape[0]):
        modified_unambiguous_cells[row][coef_idx] = -predict_ambiguous_sign[idx]


print('\n\n INCREASED ACCURACY \n\n')
label = np.ones(len(modified_ambiguous_cells))
print('MOD Ambiguous Sentence Prediction Accuracy',reg.score(modified_ambiguous_cells,label))

label = np.zeros(len(modified_unambiguous_cells))
print('MOD Unambiguous Sentence Prediction Accuracy',reg.score(modified_unambiguous_cells,label))

# END OF FILE
