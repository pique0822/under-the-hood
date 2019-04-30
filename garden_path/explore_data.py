import pandas as pd
import utils
import dictionary_corpus
import torch

model_path = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
xl = pd.read_excel('data/verb-ambiguity-with-intervening-phrase.xlsx')

with open(model_path, 'rb') as f:
    model = torch.load(f, map_location='cpu')

data_path = "../data/colorlessgreenRNNs/"


dictionary = dictionary_corpus.Dictionary(data_path)

# testing the addition of unreduced content
for i in range(len(xl)):
    break
    row = xl.iloc[i]
    base_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Ambiguous verb'].strip()

    unamb_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Unambiguous verb'].strip()

    deamb_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Unreduced content'].strip() + ' ' + row['Ambiguous verb'].strip()

    deunamb_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Unreduced content'].strip() + ' ' + row['Unambiguous verb'].strip()

    token_base = utils.tokenize_sentence(dictionary,base_sentence)
    token_unam = utils.tokenize_sentence(dictionary,unamb_sentence)
    token_deam = utils.tokenize_sentence(dictionary,deamb_sentence)
    token_deun = utils.tokenize_sentence(dictionary,deunamb_sentence)

    utils.evaluate_sentence_surprisal(model,token_base,dictionary)
    utils.evaluate_sentence_surprisal(model,token_unam,dictionary)
    utils.evaluate_sentence_surprisal(model,token_deam,dictionary)
    utils.evaluate_sentence_surprisal(model,token_deun,dictionary)

    print('\n\n\n\n')

# testing surprisal of period
memoize = {}

for i in range(len(xl)):
    # break
    row = xl.iloc[i]
    base_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Ambiguous verb'].strip() + ' ' + row['RC contents'].strip()

    alternate_sentence = row['Start'].strip() + ' ' + row['Noun'].strip() + ' ' + row['Unambiguous verb'].strip() + ' ' + row['RC contents'].strip()

    main_verb = row['Disambiguator'].strip()

    print(row['Ambiguous verb'].strip() + ' vs '+row['Unambiguous verb'].strip())

    print(base_sentence)

    largest_prob_idx, cont_surps, memoize = utils.most_likely_continuation(model, dictionary,base_sentence,['.',main_verb],memoize, return_prob = False)
    period_surp, verb_surp = cont_surps
    print(period_surp)
    print(verb_surp)
    base_processing_issue = period_surp-verb_surp

    print(alternate_sentence)
    largest_prob_idx, cont_surps, memoize = utils.most_likely_continuation(model, dictionary,alternate_sentence,['.',main_verb],memoize, return_prob = False)
    period_surp, verb_surp = cont_surps
    print(period_surp)
    print(verb_surp)
    alt_processing_issue = period_surp-verb_surp


# here is the theory, if there is some unit that has an issue that stores information about wether this is the end of the sentence or not, we should expect it to

# we will measure the score as correctly giving a period a higher surprisal than another verb (the correct verb?)
# - if this is the case then how easily will we be able detect the units? Would we need more verbs?
# - One test is to minimize the probability of "."
# - another is to maximize ( period_surp - verb_surp )
# - another is to predict that a "." is the next token from the     final hidden state. If this works, we would expect a rise in probability after the RC and at the end it should jump to 1?


# we can run thorugh each column and run one of those heatmap shits and see what comes up... The idea would be that each row/col is corresponding to one col in the df
