import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../garden_path/data/verb-ambiguity-with-intervening-phrase.csv')

ambiguous_cols = ['Start','Noun','Ambiguous verb', 'RC contents', 'Disambiguator','End']
unambiguous_cols = ['Start','Noun','Unambiguous verb', 'RC contents','Disambiguator','End']
whowas_ambiguous_cols = ['Start','Noun','Unreduced content','Ambiguous verb', 'RC contents', 'Disambiguator','End']
whowas_unambiguous_cols = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents','Disambiguator','End']

words = []
surprisals = []

with open('unmodified_gated_surprisals.txt','r') as text_file:
    for line in text_file:
        word, surprisal = line.strip().split('\t')
        words.append(word)
        surprisals.append(float(surprisal))


words_idx = -1

average_surprisals_ambiguous = [0]*len(ambiguous_cols)
average_surprisals_whowas_ambiguous = [0]*len(whowas_ambiguous_cols)
average_surprisals_unambiguous = [0]*len(unambiguous_cols)
average_surprisals_whowas_unambiguous = [0]*len(whowas_unambiguous_cols)


for ridx in range(len(data)):
    row = data.iloc[ridx]

    for cidx, col in enumerate(ambiguous_cols):
        column_words = row[col].strip()
        words_in_column = len(column_words.split(' '))

        new_words_idx = words_idx + words_in_column

        surp = 0
        if col == 'End':
            for idx in range(words_idx + 1, new_words_idx-1):
                surp += surprisals[idx]
        else:
            for idx in range(words_idx + 1, new_words_idx+1):
                surp += surprisals[idx]

        words_idx = new_words_idx
        average_surprisals_ambiguous[cidx] = (average_surprisals_ambiguous[cidx]*ridx + surp)/(ridx + 1)

    for cidx, col in enumerate(unambiguous_cols):
        column_words = row[col].strip()
        words_in_column = len(column_words.split(' '))

        new_words_idx = words_idx + words_in_column

        surp = 0
        if col == 'End':
            for idx in range(words_idx + 1, new_words_idx-1):
                surp += surprisals[idx]
        else:
            for idx in range(words_idx + 1, new_words_idx+1):
                surp += surprisals[idx]

        words_idx = new_words_idx

        average_surprisals_unambiguous[cidx] = (average_surprisals_unambiguous[cidx]*ridx + surp)/(ridx + 1)

    for cidx, col in enumerate(whowas_ambiguous_cols):
        column_words = row[col].strip()
        words_in_column = len(column_words.split(' '))

        new_words_idx = words_idx + words_in_column

        surp = 0
        if col == 'End':
            for idx in range(words_idx + 1, new_words_idx-1):
                surp += surprisals[idx]
        else:
            for idx in range(words_idx + 1, new_words_idx+1):
                surp += surprisals[idx]

        words_idx = new_words_idx
        average_surprisals_whowas_ambiguous[cidx] = (average_surprisals_whowas_ambiguous[cidx]*ridx + surp)/(ridx + 1)

    for cidx, col in enumerate(whowas_unambiguous_cols):
        column_words = row[col].strip()
        words_in_column = len(column_words.split(' '))

        new_words_idx = words_idx + words_in_column

        surp = 0
        if col == 'End':
            for idx in range(words_idx + 1, new_words_idx-1):
                surp += surprisals[idx]
        else:
            for idx in range(words_idx + 1, new_words_idx+1):
                surp += surprisals[idx]

        words_idx = new_words_idx

        average_surprisals_whowas_unambiguous[cidx] = (average_surprisals_whowas_unambiguous[cidx]*ridx + surp)/(ridx + 1)
    print('SENTENCE')

# import pdb; pdb.set_trace()

plt.ylabel('Sum Surprisal in Region')
plt.title('Averaged Sum Surprisal')
plt.plot([1,3,4,5,6],average_surprisals_ambiguous[1:],color='#0984e3',linestyle='--',label='Ambiguous Reduced')
plt.plot([1,3,4,5,6],average_surprisals_unambiguous[1:],color='#00cec9',label='Unambiguous Reduced')
plt.plot([1,2,3,4,5,6],average_surprisals_whowas_ambiguous[1:],color='#ff7675',linestyle='--',label='Ambiguous Unreduced')
plt.plot([1,2,3,4,5,6],average_surprisals_whowas_unambiguous[1:],color='#d63031',label='Unambiguous Unreduced')

plt.xticks([1,2,3,4,5,6],['Noun','Unreduced Content','RC Verb','RC Contents','Main Verb','End'], rotation=90)
plt.subplots_adjust(bottom=.250)
plt.legend()
plt.show()
