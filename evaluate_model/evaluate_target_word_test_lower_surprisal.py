# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#python3 evaluate_target_word_test.py --data ../data/colorlessgreenRNNs --checkpoint ../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt --prefixfile prefixes.txt --surprisalmode True --outf surgical_hidden.txt --modify_hidden True

#python3 evaluate_target_word_test.py --data ../data/colorlessgreenRNNs --checkpoint ../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt --prefixfile prefixes.txt --surprisalmode True --outf surgical_maximal_weight.txt --modify_cell True
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

import pandas as pd

import pickle

parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--sentences', type=int, default='-1',
                    help='number of sentences to generate from prefix')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--prefixfile', type=str, default='',
                    help='File with sentence prefix from which to generate continuations')
parser.add_argument('--surprisalmode', type=bool, default=False,
                    help='Run in surprisal mode; specify sentence with --prefixfile')
parser.add_argument('--modify_hidden', type=bool, default=False,
                    help='Modify the hidden state at the RC Verb')
parser.add_argument('--modify_cell', type=bool, default=False,
                    help='Modify the cell state at the RC Verb')
parser.add_argument('--surgical_difference', type=float, default=1.0,
                    help='constant to be used to set the cells at the RC Verb')
parser.add_argument('--file_identifier', type=str, default='',
                    help='Identifier for the .pkl files with the relevant significant units')
parser.add_argument('--gradient_type', type=bool, default='weight',
                    help='Type of gradient step to modify the x values {loss | weight}')
args = parser.parse_args()

ROOT = Path(__file__).absolute().parent.parent

significant_coef_indices = pickle.load(open(ROOT / 'garden_path'/'best_coefs'/'significant_coefs_'+args.file_identifier+'.pkl','rb+'))

best_r2_reg = np.load(open(ROOT/'garden_path'/'best_coefs'/'best_r2_coefs_'+args.file_identifier+'.pkl','rb+'))




data_df = pd.read_csv('../garden_path/data/verb-ambiguity-with-intervening-phrase.csv')

ambiguous_cols_prefix = ['Start','Noun','Ambiguous verb']
ambiguous_cols_full = ['Start','Noun','Ambiguous verb', 'RC contents']

whowas_unambiguous_cols_prefix = ['Start','Noun','Unreduced content','Unambiguous verb']
whowas_unambiguous_cols_full = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents']

modify_at_noun = ['Start','Noun']

relevant_cols = ambiguous_cols_prefix

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    print("Loading the model")
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

print("#####HERE###")

eval_batch_size = 1
seq_len = 20

dictionary = dictionary_corpus.Dictionary(args.data)
vocab_size = len(dictionary)
print("Vocab size", vocab_size)
print("TESTING")

# assuming the mask file contains one number per line indicating the index of the target word
index_col = 0


###
prefix = dictionary_corpus.tokenize(dictionary,args.prefixfile)
#print(prefix.shape)
#for w in prefix:
#    print(dictionary.idx2word[w.item()])
# try auto-generate
if not args.surprisalmode:
    print('NOT SURPRISAL')
    # print(type(prefix))
    # print(prefix.shape)
    # print(prefix)
    hidden = model.init_hidden(1)
    ntokens = dictionary.__len__()
    device = torch.device("cuda" if args.cuda else "cpu")
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    with open(args.outf, 'w') as outf:
        for i in range(args.sentences):
            for word in prefix:
                #print(word)
                #print(word.item())
                outf.write(dictionary.idx2word[word.item()] + " ")
                input.fill_(word.item())
                output, hidden = model(input,hidden)
            generated_word = None
            while generated_word != "<eos>":
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                print(word_weights)
                import pdb; pdb.set_trace()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                generated_word = dictionary.idx2word[word_idx]
                outf.write(generated_word + " ")
                output, hidden = model(input, hidden)
            outf.write("\n")

if args.surprisalmode:
#    print('SURPRISAL')
    sentences = []
    thesentence = []
    eosidx = dictionary.word2idx["<eos>"]
    for w in prefix:
#        print(w)
        thesentence.append(w)
        if w == eosidx:
            sentences.append(thesentence)
            thesentence = []
#    print(sentences)
    ntokens = dictionary.__len__()
    device = torch.device("cuda" if args.cuda else "cpu")
    with open(args.outf, 'w') as outf:
        for sidx, sentence in enumerate(sentences):

            pre_modification = ' '.join(data_df.iloc[sidx][relevant_cols])
            pre_modification = pre_modification.replace('  ',' ').lstrip().strip()

            words_to_skip = len(pre_modification.split(' '))

            torch.manual_seed(args.seed)
            hidden = model.init_hidden(1)
            input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
            totalsurprisal = 0.0
            firstword = sentence[0]
            input.fill_(firstword.item())
            outf.write(dictionary.idx2word[firstword.item()] + "\t0.00\n")
            print(dictionary.idx2word[firstword.item()] + "\t0.00\n")
            output, hidden = model(input,hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_surprisals = -1*torch.log2(word_weights/sum(word_weights))
            for widx, word in enumerate(sentence[1:len(prefix)]):
                word_surprisal = word_surprisals[word].item()
                outf.write(dictionary.idx2word[word.item()] + "\t" + str(word_surprisal) + "\n")
                print(dictionary.idx2word[word.item()] + "\t" + str(word_surprisal) + "\n")
                input.fill_(word.item())
                output, hidden = model(input, hidden)
                  # widx == 1 means we are at the RC verb
                  # We only want to modify if the args say so
                  # every fourth sentence starting with the first is ambiguous
                if widx == words_to_skip-2 and args.modify_cell:
                    print('MODIFY HERE',dictionary.idx2word[word.item()] )

                    for coef_idx in range(len(significant_coef_indices)):
                        modified_unit = significant_coef_indices[coef_idx]

                        if args.gradient_type == 'loss':
                            gradient = 2*(0 - best_r2_reg.coef_[modified_unit]*hidden[1][1][0,modified_unit] - best_r2_reg.intercept_)*(-best_r2_reg.coef_[modified_unit])

                        elif args.gradient_type == 'weight':
                            gradient = best_r2_reg.coef_[modified_unit]

                        hidden[1][1][0,modified_unit] = hidden[1][1][0,modified_unit] -args.surgical_difference *float(gradient)


                  word_weights = output.squeeze().div(args.temperature).exp().cpu()
                  word_surprisals = -1*torch.log2(word_weights/sum(word_weights))
