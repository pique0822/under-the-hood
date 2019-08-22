import argparse
from pathlib import Path

import pandas as pd
import yaml


def main(args):
    experiment = yaml.load(args.experiment_file)["experiment"]
    stimuli_path = Path(args.experiment_file.name).parent / experiment["stimuli"]
    print(stimuli_path)

    stimuli = pd.read_csv(stimuli_path)
    for _, row in stimuli.iterrows():
        for condition in experiment["conditions"].values():
            prefix = [row[region].strip() for region in condition["prefix_columns"]]
            # TODO NB hard-coded <eos>
            sentence = prefix + [row[condition["measure_column"]].strip(), "<eos>"]
            sentence = " ".join(sentence)

            args.outf.write(sentence + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment_file", type=argparse.FileType("r"))
    p.add_argument("--outf", type=argparse.FileType("w"))

    args = p.parse_args()
    main(args)
