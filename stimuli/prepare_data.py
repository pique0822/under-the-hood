from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

p = ArgumentParser()
p.add_argument("csv_dir", type=Path)

args = p.parse_args()


csv_files = list(args.csv_dir.glob("*.csv"))
concatenated = pd.concat([pd.read_csv(csv_file, index_col=0) for csv_file in csv_files],
                         keys=[csv_file.stem for csv_file in csv_files],
                         names=["source", "idx"])
concatenated.to_csv("%s.csv" % (args.csv_dir.name))
