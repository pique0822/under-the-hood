import argparse
from pathlib import Path

import pandas as pd
import yaml

from experiment_util import Experiment


def main(args):
    experiment = Experiment.from_yaml(args.experiment_file)
    for sentence in experiment.get_sentences():
        args.outf.write(sentence.text + "\n")

        if args.extract_idx_outf:
            args.extract_idx_outf.write("%i\n" % sentence.extract_idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment_file", type=Path)
    p.add_argument("--outf", type=argparse.FileType("w"))
    p.add_argument("--extract_idx_outf", type=argparse.FileType("w"),
            help=("If provided, write a single integer per line "
                  "indicating zero-indexed position of extract column's "
                  " final token in the corresponding sentence."))

    args = p.parse_args()
    main(args)
