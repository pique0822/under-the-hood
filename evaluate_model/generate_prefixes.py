import argparse
from pathlib import Path

import pandas as pd
import yaml


def main(args):
    experiment = yaml.load(args.experiment_file)["experiment"]
    stimuli_path = Path(args.experiment_file.name).parent / experiment["stimuli"]

    stimuli = pd.read_csv(stimuli_path)
    for _, row in stimuli.iterrows():
        for condition in experiment["conditions"].values():
            prefix = []
            token_idx = 0
            extract_column_token_idx = None
            for region in condition["prefix_columns"]:
                region_tokens = row[region].strip().split(" ")

                token_idx += len(region_tokens)
                if region == condition["extract_column"]:
                    extract_column_token_idx = token_idx - 1

                prefix.extend(region_tokens)

            # TODO NB hard-coded <eos>
            sentence = prefix + [row[condition["measure_column"]].strip(), "<eos>"]
            sentence = " ".join(sentence)

            args.outf.write(sentence + "\n")

            if args.extract_idx_outf:
                args.extract_idx_outf.write("%i\n" % extract_column_token_idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment_file", type=argparse.FileType("r"))
    p.add_argument("--outf", type=argparse.FileType("w"))
    p.add_argument("--extract_idx_outf", type=argparse.FileType("w"),
            help=("If provided, write a single integer per line "
                  "indicating zero-indexed position of extract column's "
                  " final token in the corresponding sentence."))

    args = p.parse_args()
    main(args)
