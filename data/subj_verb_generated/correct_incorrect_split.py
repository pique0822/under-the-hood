import dictionary_corpus
import numpy as np
import torch

def tokenize_sentence(dictionary, line):
    """Tokenizes one line at a time
    dictionary is a Dictionary as defined in dictionary_corpus.
    """
    words = line.split()
    ntokens = len(words)

    ids = torch.LongTensor(ntokens)
    token = 0
    for word in words:
        if word in dictionary.word2idx:
            ids[token] = dictionary.word2idx[word]
        else:
            ids[token] = dictionary.word2idx["<unk>"]
        token += 1
    return ids


def get_probability_vector(dictionary, sentence):
    prefix = tokenize_sentence(dictionary, sentence)

    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    firstword = prefix[0]
    input.fill_(firstword.item())
    output, hidden = model(input,hidden)
    word_weights = output.squeeze().div(1).exp().cpu()
    probs = word_weights/sum(word_weights)

    for word in prefix[1:len(prefix)]:
        prob = probs[word].item()
        input.fill_(word.item())
        output, hidden = model(input,hidden)
        word_weights = output.squeeze().div(1).exp().cpu()
        probs = word_weights/sum(word_weights)

    return probs

def most_likely_continuation(dictionary, prefix, possible_conts, memoize = {}):
    if prefix in memoize:
        probs = memoize[prefix]
    else:
        probs = get_probability_vector(dictionary, prefix)
        memoize[prefix] = probs
    tokenized_conts = []
    for continuation in possible_conts:
        tokenized_conts.append(tokenize_sentence(dictionary, continuation))

    cont_probs = []
    largest_prob = 0
    largest_prob_idx = 0
    for idx, tconts in enumerate(tokenized_conts):
        prob = probs[tconts].item()
        cont_probs.append(prob)

        if prob > largest_prob:
            largest_prob = prob
            largest_prob_idx = idx

    return largest_prob_idx, cont_probs, memoize

def most_likely_continuation_lists(dictionary, prefix, list_of_possible_conts):
    probs = get_probability_vector(dictionary, prefix)
    indices = []
    for possible_conts in list_of_possible_conts:
        tokenized_conts = []
        for continuation in possible_conts:
            tokenized_conts.append(tokenize_sentence(dictionary, continuation))

        cont_probs = []
        largest_prob = 0
        largest_prob_idx = 0
        for idx, tconts in enumerate(tokenized_conts):
            prob = probs[tconts].item()
            cont_probs.append(prob)

            if prob > largest_prob:
                largest_prob = prob
                largest_prob_idx = idx

        indices.append(largest_prob_idx)
    return indices

model_file = '../../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
data = '../colorlessgreenRNNs/'
cuda = False

print('LOAD MODEL...')
with open(model_file, 'rb') as f:
    print("Loading the model")
    if cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)

print('SET UP FILES...')
dictionary = dictionary_corpus.Dictionary(data)
vocab_size = len(dictionary)
ntokens = dictionary.__len__()
device = torch.device("cuda" if cuda else "cpu")

# get all verbs
sing_verb_file = 'verbs/long_singular_verbs.txt'
plu_verb_file = 'verbs/long_plural_verbs.txt'

sing_verbs = []
plu_verbs = []

with open(sing_verb_file) as verb_file:
    for verb in verb_file:
        sing_verbs.append(verb.strip())

with open(plu_verb_file) as verb_file:
    for verb in verb_file:
        plu_verbs.append(verb.strip())


dataset_file = 'datasets/subj_verb_generated_dataset_3'
# The sentence will then be of the form
# 'the ____ that had broken the vase _____ frequently'

print('COMPARE VERBS...')
correct = []
incorrect = []
ct = 0

memoize_prefix_prob = {}
with open(dataset_file+'.txt','r') as data:
    for line in data:
        agreement, singular, sentence = line.split('\t')
        agreement = int(agreement)
        if agreement == 0:
            continue
        singular = int(singular)
        sentence = sentence.strip()
        prefix = ' '.join(sentence.split(' ')[:7])

        sentence_verb = sentence.split(' ')[7]
        if (agreement and singular) or (not agreement and not singular):
            vb_idx = sing_verbs.index(sentence_verb)
        elif (agreement and not singular) or (not agreement and singular):
            vb_idx = plu_verbs.index(sentence_verb)

        singular_verb = sing_verbs[vb_idx]
        plural_verb = plu_verbs[vb_idx]

        possible_conts = [plural_verb, singular_verb]

        prob_idx, _, memoize_prefix_prob = most_likely_continuation(dictionary, prefix, possible_conts, memoize_prefix_prob)

        # if prob_idx == singular then the model agrees with the sentence
        # otherwise it would incorrectly classify them
        if int(prob_idx) == int(singular):
            correct.append(line)
        else:
            incorrect.append(line)
        ct += 1


print('CORRECT',len(correct))
print('incorrect',len(incorrect))

print('WRITE TO FILES...')
with open(dataset_file+'_correct.txt','w+') as file:
    for line in correct:
        file.write(line)

with open(dataset_file+'_incorrect.txt','w+') as file:
    for line in incorrect:
        file.write(line)
