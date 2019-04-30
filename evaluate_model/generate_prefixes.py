import pandas as pd

data = pd.read_csv('../garden_path/data/verb-ambiguity-with-intervening-phrase.csv')

ambiguous_cols = ['Start','Noun','Ambiguous verb', 'RC contents', 'Intervener','Disambiguator','End']
whowas_ambiguous_cols = ['Start','Noun','Unreduced content','Ambiguous verb', 'RC contents', 'Intervener','Disambiguator','End']
unambiguous_cols = ['Start','Noun','Unambiguous verb', 'RC contents', 'Intervener','Disambiguator','End']
whowas_unambiguous_cols = ['Start','Noun','Unreduced content','Unambiguous verb', 'RC contents', 'Intervener','Disambiguator','End']

with open('prefixes.txt','w') as prefix_file:
    for r_idx in range(len(data)):
        row = data.iloc[r_idx]

        ambiguous_sentence = ""
        for col in ambiguous_cols:
            ambiguous_sentence += ' ' + row[col].strip().lstrip()
            ambiguous_sentence = ambiguous_sentence.lstrip()

        whowas_ambiguous_sentence = ""
        for col in whowas_ambiguous_cols:
            whowas_ambiguous_sentence += ' ' + row[col].strip().lstrip()
            whowas_ambiguous_sentence = whowas_ambiguous_sentence.lstrip()

        unambiguous_sentence = ""
        for col in unambiguous_cols:
            unambiguous_sentence += ' ' + row[col].strip().lstrip()
            unambiguous_sentence = unambiguous_sentence.lstrip()

        whowas_unambiguous_sentence = ""
        for col in whowas_unambiguous_cols:
            whowas_unambiguous_sentence += ' ' + row[col].strip().lstrip()
            whowas_unambiguous_sentence = whowas_unambiguous_sentence.lstrip()

        prefix_file.write(ambiguous_sentence+'\n')
        prefix_file.write(whowas_ambiguous_sentence+'\n')
        prefix_file.write(unambiguous_sentence+'\n')
        prefix_file.write(whowas_unambiguous_sentence+'\n')
