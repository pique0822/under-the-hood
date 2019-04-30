import torch
import numpy as np

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


def get_probability_vector(model, dictionary, sentence):
    ntokens = dictionary.__len__()
    prefix = tokenize_sentence(dictionary, sentence)

    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')

    firstword = prefix[0]
    input.fill_(firstword.item())
    output, hidden = model(input,hidden)
    # import pdb; pdb.set_trace()
    word_weights = output.squeeze().div(1).exp().cpu()
    probs = word_weights/sum(word_weights)
    # import pdb; pdb.set_trace()
    for word in prefix[1:len(prefix)]:
        prob = probs[word].item()
        input.fill_(word.item())
        output, hidden = model(input,hidden)
        word_weights = output.squeeze().div(1).exp().cpu()
        probs = word_weights/sum(word_weights)

    return probs

def most_likely_continuation(model, dictionary, prefix, possible_conts, memoize = {}, return_prob = True):
    if prefix in memoize:
        probs = memoize[prefix]
    else:
        probs = get_probability_vector(model, dictionary, prefix)
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

    cont_probs
    if return_prob == False:
        for i in range(len(cont_probs)):
            p = cont_probs[i]
            s = probability_to_surprisal(p)
            cont_probs[i] = s

    return largest_prob_idx, cont_probs, memoize

def most_likely_continuation_lists(model, dictionary, prefix, list_of_possible_conts):
    probs = get_probability_vector(model, dictionary, prefix)
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

def evaluate_sentence_surprisal(model,prefix,dictionary):
    sentences = []
    thesentence = []
    eosidx = dictionary.word2idx["<eos>"]
    for w in prefix:
        # print(w)
        thesentence.append(w)
        if w == eosidx:
            sentences.append(thesentence)
            thesentence = []
    sentences.append(thesentence)
    # print(sentences)
    ntokens = dictionary.__len__()
    for sentence in sentences:
        torch.manual_seed(1111)
        hidden = model.init_hidden(1)
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')
        totalsurprisal = 0.0
        firstword = sentence[0]
        input.fill_(firstword.item())
        print(dictionary.idx2word[firstword.item()] + "\t0.00\n")
        output, hidden = model(input,hidden)
        word_weights = output.squeeze().div(1).exp().cpu()
        word_surprisals = -1*torch.log2(word_weights/sum(word_weights))
        for word in sentence[1:len(prefix)]:
              word_surprisal = word_surprisals[word].item()
              print(dictionary.idx2word[word.item()] + "\t" + str(word_surprisal) + "\n")
              input.fill_(word.item())
              output, hidden = model(input, hidden)
              word_weights = output.squeeze().div(1).exp().cpu()
              word_surprisals = -1*torch.log2(word_weights/sum(word_weights))

def probability_to_surprisal(prob):
    return -np.log2(prob)

def surprisal_to_probability(surp):
    return 2**(-surp)
