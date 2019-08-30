import argparse
from collections import defaultdict
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from scipy import stats


parser = argparse.ArgumentParser(description='Suprisal plot generator')
parser.add_argument("experiment_file", type=Path)
parser.add_argument("surprisal_file", type=Path)
parser.add_argument("--surgical_files", type=Path, nargs="+")
parser.add_argument('--file_title', type=str, default = '',
                    help='Title to be used for plot title specification')
parser.add_argument('--save_file', type=str, default = 'surprisal_plots',
                    help='file to be used for plot saving')
args = parser.parse_args()

# IGNORE WARNINGS
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
######

experiment = yaml.load(args.experiment_file)["experiment"]

stimuli_path = Path(args.experiment_file.name).parent / experiment["stimuli"]
data = pd.read_csv(stimuli_path, header=0, index_col=0).sort_index()
surp_df = pd.read_csv(args.surprisal_file, header=0, index_col=["sentence_id", "token_id"], sep="\t").sort_index()


# List of (sentence_id, condition, model_spec, region, surprisal)
graph_data = []

# Make a map from stimulus ID and condition to sentence ID as it will appear in
# surprisal data.
#
# TODO make sure ordering is stable from YAML loader .. or better, just specify
# a list in the YAML :)
items = [(idx, condition)
         for idx in stimuli.index
         for condition in experiment["conditions"].keys()]
sentence_to_item = {i + 1: item for i, item in enumerate(items)}


# First aggregate surprisal data from the base model.
for sentence_id, surprisals in surp_df.groupby("sentence_id"):
    item_idx, condition = surp_df.loc[sentence_to_item[sentence_id]]
    item = surp_df.loc[item_idx]

    i = 0
    for region in experiment["conditions"][condition]["prefix_columns"]:
        region_tokens = item[region].strip().split(" ")
        region_surprisals = surprisals[i:i + len(region_tokens)]
        assert len(region_tokens) == len(region_surprisals)

        graph_data.append((sentence_id, condition, "baseline", sum(region_surprisals)))

# Now add surprisal data from the surgery models.
for surgical_file in args.surgical_files:
    with surgical_file.open("rb") as surgical_f:
        surgical_data = pickle.load(surgical_f)
        # TODO Read surgery coef and add to loop above
        key = "surgery_%f" % surgical_data["surgery_coef"]


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


ambiguous_cols = ['Start','Noun','Ambiguous verb', 'RC contents', 'Disambiguator','End']
whowas_ambiguous_cols = ['Start','Noun','Unreduced content','Ambiguous verb', 'RC contents', 'Disambiguator','End']
unambiguous_cols = ['Start','Noun','Unambiguous verb', 'RC contents','Disambiguator','End']
whowas_unambiguous_cols = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents','Disambiguator','End']



ambiguous_cols_prefix = ['Start','Noun','Ambiguous verb', 'RC contents']
whowas_ambiguous_cols_prefix = ['Start','Noun','Unreduced content','Ambiguous verb', 'RC contents']
unambiguous_cols_prefix = ['Start','Noun','Unambiguous verb', 'RC contents']
whowas_unambiguous_cols_prefix = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents']

words = []
surprisals = []
surprisal_data = {'Time':[],'Avg Surprisal':[],'File idx':[]}
flatui = ["#00cec9", "#CC7675","#00ceFF", "#ff7675"]
sns.set_palette(flatui)

with open('gold_standard.txt','r') as text_file:
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



        for cidx, col in enumerate(ambiguous_cols):
            column_words = row[col].strip()
            words_in_column = len(column_words.split(' '))

            new_words_idx = words_idx + words_in_column

            surp = 0
            if col == 'End':
                for idx in range(words_idx + 1, new_words_idx-1):
                    surp += surprisals[idx]
            else:
                if col is 'Disambiguator' and ambiguous_prefix in prefix_to_avg:
                    surp+= prefix_to_avg[ambiguous_prefix]
                else:
                    for idx in range(words_idx + 1, new_words_idx+1):
                        surp += surprisals[idx]

            words_idx = new_words_idx
            average_surprisals_ambiguous[cidx] = (average_surprisals_ambiguous[cidx]*ridx + surp)/(ridx + 1)

            if (col is not 'Start' and col is not 'Disambiguator') or (col is 'Disambiguator' and ambiguous_prefix not in prefix_to_avg):
                if 'verb' in col:
                    surprisal_data['Time'].append('Verb')
                else:
                    surprisal_data['Time'].append(col)
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(0)
            elif (col is 'Disambiguator' and ambiguous_prefix in prefix_to_avg):
                surprisal_data['Time'].append(col)
                surp = prefix_to_avg[ambiguous_prefix]
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(0)

        for cidx, col in enumerate(whowas_ambiguous_cols):
            column_words = row[col].strip()
            words_in_column = len(column_words.split(' '))

            new_words_idx = words_idx + words_in_column

            surp = 0
            if col == 'End':
                for idx in range(words_idx + 1, new_words_idx-1):
                    surp += surprisals[idx]
            else:
                if col is 'Disambiguator' and whowas_ambiguous_prefix in prefix_to_avg:
                    surp+= prefix_to_avg[whowas_ambiguous_prefix]
                else:
                    for idx in range(words_idx + 1, new_words_idx+1):
                        surp += surprisals[idx]

            words_idx = new_words_idx
            average_surprisals_whowas_ambiguous[cidx] = (average_surprisals_whowas_ambiguous[cidx]*ridx + surp)/(ridx + 1)

            if (col is not 'Start' and col is not 'Disambiguator') or (col is 'Disambiguator' and whowas_ambiguous_prefix not in prefix_to_avg):
                if 'verb' in col:
                    surprisal_data['Time'].append('Verb')
                else:
                    surprisal_data['Time'].append(col)
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(1)
            elif (col is 'Disambiguator' and whowas_ambiguous_prefix in prefix_to_avg):
                surprisal_data['Time'].append(col)
                surp = prefix_to_avg[whowas_ambiguous_prefix]
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(1)

        for cidx, col in enumerate(unambiguous_cols):
            column_words = row[col].strip()
            words_in_column = len(column_words.split(' '))

            new_words_idx = words_idx + words_in_column

            surp = 0
            if col == 'End':
                for idx in range(words_idx + 1, new_words_idx-1):
                    surp += surprisals[idx]
            else:
                if col is 'Disambiguator' and unambiguous_prefix in prefix_to_avg:
                    surp+= prefix_to_avg[unambiguous_prefix]
                else:
                    for idx in range(words_idx + 1, new_words_idx+1):
                        surp += surprisals[idx]

            words_idx = new_words_idx

            average_surprisals_unambiguous[cidx] = (average_surprisals_unambiguous[cidx]*ridx + surp)/(ridx + 1)

            if (col is not 'Start' and col is not 'Disambiguator') or (col is 'Disambiguator' and unambiguous_prefix not in prefix_to_avg):
                if 'verb' in col:
                    surprisal_data['Time'].append('Verb')
                else:
                    surprisal_data['Time'].append(col)
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(2)
            elif (col is 'Disambiguator' and unambiguous_prefix in prefix_to_avg):
                surprisal_data['Time'].append(col)
                surp = prefix_to_avg[unambiguous_prefix]
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(2)

        for cidx, col in enumerate(whowas_unambiguous_cols):
            column_words = row[col].strip()
            words_in_column = len(column_words.split(' '))

            new_words_idx = words_idx + words_in_column

            surp = 0
            if col == 'End':
                for idx in range(words_idx + 1, new_words_idx-1):
                    surp += surprisals[idx]
            else:
                if col is 'Disambiguator' and whowas_unambiguous_prefix in prefix_to_avg:
                    surp+= prefix_to_avg[whowas_unambiguous_prefix]
                else:
                    for idx in range(words_idx + 1, new_words_idx+1):
                        surp += surprisals[idx]

            words_idx = new_words_idx

            average_surprisals_whowas_unambiguous[cidx] = (average_surprisals_whowas_unambiguous[cidx]*ridx + surp)/(ridx + 1)

            if (col is not 'Start' and col is not 'Disambiguator') or (col is 'Disambiguator' and whowas_unambiguous_prefix not in prefix_to_avg):
                if 'verb' in col:
                    surprisal_data['Time'].append('Verb')
                else:
                    surprisal_data['Time'].append(col)
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(3)
            elif (col is 'Disambiguator' and whowas_unambiguous_prefix in prefix_to_avg):
                surprisal_data['Time'].append(col)
                surp = prefix_to_avg[whowas_unambiguous_prefix]
                surprisal_data['Avg Surprisal'].append(surp)
                surprisal_data['File idx'].append(3)

plt.plot([0,2,3,4,5],average_surprisals_ambiguous[1:],linestyle='--',label='Ambiguous Reduced True',color=flatui[0])

plt.ylabel('Sum Surprisal in Region')
plt.plot([0,2,3,4,5],average_surprisals_unambiguous[1:],color=flatui[2],label='Unambiguous Reduced')
plt.plot([0,1,2,3,4,5],average_surprisals_whowas_ambiguous[1:],color=flatui[1],linestyle='--',label='Ambiguous Unreduced')
plt.plot([0,1,2,3,4,5],average_surprisals_whowas_unambiguous[1:],color=flatui[3],label='Unambiguous Unreduced')

gold_surp_df = pd.DataFrame.from_dict(surprisal_data)
ax = sns.pointplot(x='Time',y='Avg Surprisal', hue='File idx',join=False,data=gold_surp_df,order=['Noun','Unreduced content','Verb','RC contents','Disambiguator','End'], legend=False,label='_nolegend_',dodge=True)
weight_test = []


### START DECREASE ###
if args.surgical_decrease:
    surprisal_data = {'Time':[],'Avg Surprisal':[],'File idx':[]}
    flatui = ["#EE5A24", "#009432", "#0652DD", "#9980FA"]
    sns.set_palette(flatui)
    weight_test = args.decrease_file_unique_ids
    for hue,file_name_idx in enumerate(weight_test):

        words = []
        surprisals = []
        # print('surgical_cell_verb_'+str(file_name_idx)+'.txt')
        with open(args.decrease_file_base+str(file_name_idx)+'.txt','r') as text_file:
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

                if col is not 'Start':
                    if 'verb' in col:
                        surprisal_data['Time'].append('Verb')
                    else:
                        surprisal_data['Time'].append(col)
                    surprisal_data['Avg Surprisal'].append(surp)
                    surprisal_data['File idx'].append(hue)

                words_idx = new_words_idx
                average_surprisals_ambiguous[cidx] = (average_surprisals_ambiguous[cidx]*ridx + surp)/(ridx + 1)

        plt.plot([0,2,3,4,5],average_surprisals_ambiguous[1:],linestyle='--',label='Ambiguous Reduced '+str(file_name_idx),color=flatui[hue])

    surp_df = pd.DataFrame.from_dict(surprisal_data)
    ax = sns.pointplot(x='Time',y='Avg Surprisal', hue='File idx',data=surp_df,order=['Noun','Unreduced content','Verb','RC contents','Disambiguator','End'],join=False, legend=False,label='_nolegend_',dodge=True)
    ### END DECREASE ###
    pos_locs = ['Noun','Verb','RC contents','Disambiguator','End']
    print('\n\nSignificance Measures:')
    for hue, surp_df_hue in surp_df.groupby('File idx'):
        print('\n\n')
        for pos in pos_locs:

            surprisal_list = surp_df_hue[surp_df_hue['Time'] == pos]['Avg Surprisal'].tolist()

            # if hue == 2 and pos == 'Disambiguator':
            #     import pdb; pdb.set_trace()

            golden_surprisal_list = gold_surp_df[gold_surp_df['Time'] == pos] [gold_surp_df['File idx'] == 0]['Avg Surprisal'].tolist()
            print('Value : '+str(weight_test[hue])+' at Site : '+str(pos)+' -- p_value = ',stats.ttest_rel(golden_surprisal_list,surprisal_list).pvalue)




# ### START INCREASE ###
# surprisal_data = {'Time':[],'Avg Surprisal':[],'File idx':[]}
# flatui = ["#f8c291", "#e55039", "#eb2f06", "#b71540"]
# sns.set_palette(flatui)
#
# for hue,file_name_idx in enumerate([0.1,1,10]):
#     plt.title('Surprisal Increase on Unambiguous Unreduced')
#
#     words = []
#     surprisals = []
#     # print('surgical_cell_verb_'+str(file_name_idx)+'.txt')
#     with open('surgical_gradient_r2_increase_'+str(file_name_idx)+'.txt','r') as text_file:
#         for line in text_file:
#             word, surprisal = line.strip().split('\t')
#             words.append(word)
#             surprisals.append(float(surprisal))
#
#
#     words_idx = -1
#
#     average_surprisals_ambiguous = [0]*len(ambiguous_cols)
#     average_surprisals_whowas_ambiguous = [0]*len(whowas_ambiguous_cols)
#     average_surprisals_unambiguous = [0]*len(unambiguous_cols)
#     average_surprisals_whowas_unambiguous = [0]*len(whowas_unambiguous_cols)
#
#
#     for ridx in range(len(data)):
#         row = data.iloc[ridx]
#
#         for cidx, col in enumerate(whowas_unambiguous_cols):
#             column_words = row[col].strip()
#             words_in_column = len(column_words.split(' '))
#
#             new_words_idx = words_idx + words_in_column
#
#             surp = 0
#             if col == 'End':
#                 for idx in range(words_idx + 1, new_words_idx-1):
#                     surp += surprisals[idx]
#             else:
#                 for idx in range(words_idx + 1, new_words_idx+1):
#                     surp += surprisals[idx]
#
#             words_idx = new_words_idx
#
#             average_surprisals_whowas_unambiguous[cidx] = (average_surprisals_whowas_unambiguous[cidx]*ridx + surp)/(ridx + 1)
#
#             if col is not 'Start':
#                 if 'verb' in col:
#                     surprisal_data['Time'].append('Verb')
#                 else:
#                     surprisal_data['Time'].append(col)
#                 surprisal_data['Avg Surprisal'].append(surp)
#                 surprisal_data['File idx'].append(hue)
#
#     plt.plot([0,1,2,3,4,5],average_surprisals_whowas_unambiguous[1:],label='Unambiguous Unreduced '+str(file_name_idx),color=flatui[hue])
#
# surp_df = pd.DataFrame.from_dict(surprisal_data)
# ax = sns.pointplot(x='Time',y='Avg Surprisal', hue='File idx',data=surp_df,order=['Noun','Unreduced content','Verb','RC contents','Disambiguator','End'],join=False)
# ### END INCREASE ###

plt.title('Averaged VBD Surprisal Plots :: '+args.file_title)
handles, labels = ax.get_legend_handles_labels()

plt.legend(handles[0:4+len(weight_test)], labels[0:4+len(weight_test)])
# plt.subplots_adjust(right=0.6)
plt.tick_params( labelsize='small', labelrotation=45)

plt.tight_layout()
plt.savefig(args.save_file+'.png')
