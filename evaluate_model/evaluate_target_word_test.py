# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
from pathlib import Path
import pickle
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

L = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
parser.add_argument('--outf', type=argparse.FileType("w"), default=None,
                    help='output file for generated text')
parser.add_argument('--prefixfile', type=str, default='-',
                    help='File with sentence prefix from which to generate continuations')
parser.add_argument('--surprisalmode', type=bool, default=False,
                    help='Run in surprisal mode; specify sentence with --prefixfile')

parser.add_argument("--do_surgery", type=bool, default=False)
parser.add_argument("--surgery_idx_file", type=Path)
parser.add_argument("--surgery_coef_file", type=Path)
parser.add_argument("--gradient_type", choices=["loss", "weight"], default="weight")
parser.add_argument("--surgery_scale", type=float, default=1.0)
parser.add_argument("--surgery_outf", type=Path)


args = parser.parse_args()

# prepare surgery data if necessary
if args.do_surgery:
    # surgery idxs: specifies the token index at which to perform surgery for
    # each sentence
    with args.surgery_idx_file.open("r") as idx_f:
        surgery_idxs = [int(line.strip()) for line in idx_f if line.strip()]

    with args.surgery_coef_file.open("rb") as coef_f:
        surgery_decoder = pickle.load(coef_f)


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        sys.stderr.write("WARNING: You have a CUDA device, so you should probably run with --cuda\n")
    else:
        torch.cuda.manual_seed(args.seed)

L.info("Loading language model from checkpoint %s", args.checkpoint)
with open(args.checkpoint, 'rb') as f:
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

dictionary = dictionary_corpus.Dictionary(args.data)
vocab_size = len(dictionary)

L.info("Model ready.")


def do_surgery(hidden, decoder, scale=1.0):
    """
    Do surgery in-place on the hidden state of the LSTM.
    """
    if args.gradient_type == "weight":
        gradient = decoder.coef_
    elif args.gradient_type == "loss":
        raise ValueError("not supported")

    # TODO don't recreate tensor every time
    hidden[1][1][0, :] -= scale * torch.tensor(gradient)
    return hidden


###
prefix = dictionary_corpus.tokenize(dictionary, args.prefixfile)
L.info("Tokenization complete.")
#print(prefix.shape)
#for w in prefix:
#    print(dictionary.idx2word[w.item()])
# try auto-generate
if not args.surprisalmode:
    # print(type(prefix))
    # print(prefix.shape)
    # print(prefix)
    hidden = model.init_hidden(1)
    ntokens = dictionary.__len__()
    device = torch.device("cuda" if args.cuda else "cpu")
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    with args.outf as outf:
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
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                generated_word = dictionary.idx2word[word_idx]
                outf.write(generated_word + " ")
                output, hidden = model(input, hidden)
            outf.write("\n")


if args.surprisalmode:
    sentences = []
    thesentence = []
    eosidx = dictionary.word2idx["<eos>"]
    for w in prefix:
        thesentence.append(w)
        if w == eosidx:
            sentences.append(thesentence)
            thesentence = []

    if args.do_surgery:
        assert len(sentences) == len(surgery_idxs), "Mismatched input / surgery instructions"

    ntokens = dictionary.__len__()
    device = torch.device("cuda" if args.cuda else "cpu")

    results = []

    with torch.no_grad():
        for i, sentence in tqdm(list(enumerate(sentences))):
            # Prepare for surgery.
            surgery_idx = surgery_idxs[i] if args.do_surgery else None

            torch.manual_seed(args.seed)
            hidden = model.init_hidden(1)
            input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
            totalsurprisal = 0.0
            firstword = sentence[0]
            input.fill_(firstword.item())

            results.append((i + 1, 1, dictionary.idx2word[firstword.item()], 0.))

            output, hidden = model(input,hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_surprisals = -1*torch.log2(word_weights/sum(word_weights))
            for j, word in enumerate(sentence[1:]):
                if j + 1 == surgery_idx:
                    # Perform surgery on hidden state.
                    L.info("Performing surgery for sentence %i at token: %s",
                            i, dictionary.idx2word[word.item()])
                    pre_hidden_norm = torch.norm(hidden[1][1][0]).item()
                    hidden = do_surgery(hidden, surgery_decoder, args.surgery_scale)
                    post_hidden_norm = torch.norm(hidden[1][1][0]).item()
                    L.info("Norm change: %s -> %s", pre_hidden_norm, post_hidden_norm)

                word_surprisal = word_surprisals[word].item()
                results.append((i + 1, j + 2, dictionary.idx2word[word.item()], word_surprisal))
                input.fill_(word.item())
                output, hidden = model(input, hidden)

                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_surprisals = -1*torch.log2(word_weights/sum(word_weights))

    results = pd.DataFrame(results, columns=["sentence_id", "token_id", "token", "surprisal"])

    if args.outf is not None:
        results.to_csv(args.outf, sep="\t")
    if args.surgery_outf is not None:
        with args.surgery_outf.open("wb") as surgery_f:
            pickle.dump({"surgery_coef": args.surgery_scale, "results": results}, surgery_f)
