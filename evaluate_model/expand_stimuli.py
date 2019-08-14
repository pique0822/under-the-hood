from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def main(args):
    data = pd.read_csv(args.stimuli_file)
    items = pd.read_csv(args.corpus_stats_file)
    items = items[items.train_count >= args.frequency_threshold]

    data_dict = {'sentence': [], 'ambiguous': [], 'reduced': [], 'post_rc_POS': []}
    for r_idx in range(len(data)):
        row = data.iloc[r_idx]

        for ambiguous in [True, False]:
            for reduced in [True, False]:
                sentence = row['Start'] + ' ' + row['Noun'] + ' '
                if not reduced:
                    sentence += row['Unreduced content'] + ' '
                if ambiguous:
                    sentence += row['Ambiguous verb'] + ' '
                else:
                    sentence += row['Unambiguous verb'] + ' '
                sentence += row['RC contents'] + ' ' # + row['Intervener'] + ' '

                for r_item_idx in range(len(items)):
                    item_row = items.iloc[r_item_idx]
                    data_dict['sentence'].append(sentence + item_row['token'].lower() + ' <eos>')
                    data_dict['ambiguous'].append(ambiguous)
                    data_dict['reduced'].append(reduced)
                    data_dict['post_rc_POS'].append(item_row['POS'])

    with args.out_file.open("w") as out_f:
        df.to_csv(out_f, index=False)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("stimuli_file", type=Path)
    p.add_argument("corpus_stats_file", type=Path)
    p.add_argument("out_file", type=Path)
    p.add_argument("--frequency_threshold", default=100, type=int)

    args = p.parse_args()
    main(args)
