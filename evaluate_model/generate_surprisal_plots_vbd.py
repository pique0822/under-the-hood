import argparse
from collections import defaultdict
from pathlib import Path
import pickle

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from util import Experiment, read_surprisal_df


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

experiment = Experiment.from_yaml(args.experiment_file)

surp_dfs = {
    "baseline": read_surprisal_df(args.surprisal_file),
}

# Now add surprisal data from the surgery models.
for surgical_file in args.surgical_files:
    with surgical_file.open("rb") as surgical_f:
        surgical_data = pickle.load(surgical_f)
        # TODO Read surgery coef and add to loop above
        key = "surgery_%f" % surgical_data["surgery_coef"]
        surp_dfs[key] = surgical_data["results"]


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


# Aggregate surprisal data into a single graph-friendly dataframe.
for model_key, surp_df in surp_dfs.items():
    for sentence_id, surprisals in surp_df.groupby("sentence_id"):
        item_idx, condition = surp_df.loc[sentence_to_item[sentence_id]]
        item = surp_df.loc[item_idx]

        i = 0
        for region in experiment["conditions"][condition]["prefix_columns"]:
            region_tokens = item[region].strip().split(" ")
            region_surprisals = surprisals[i:i + len(region_tokens)]
            assert len(region_tokens) == len(region_surprisals)

            graph_data.append((sentence_id, condition, model_key, region, sum(region_surprisals)))


graph_data = pd.DataFrame(graph_data, columns=["sentence_id", "condition", "model", "region", "sum_surprisal"])
# Now map regions to cross-condition time-points.
graph_times = experiment["plot"]["times"]
region_to_time = {region: time for region in time_regions
                  for time, time_regions in graph_times.items()}
from pprint import pprint
pprint(region_to_time)

graph_data["time"] = graph_data.region.transform(lambda x: region_to_time[x])

# TODO graph line plots.
