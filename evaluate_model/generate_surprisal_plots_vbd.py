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

from experiment_util import Experiment, read_surprisal_df


parser = argparse.ArgumentParser(description='Suprisal plot generator')
parser.add_argument("experiment_file", type=Path)
parser.add_argument("surprisal_file", type=Path)
parser.add_argument("--surgery_files", type=Path, nargs="+")
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
for surgical_file in args.surgery_files:
    with surgical_file.open("rb") as surgical_f:
        surgical_data = pickle.load(surgical_f)
        # Read surgery coef and add to loop above
        key = "surgery_%f" % surgical_data["surgery_coef"]
        surp_dfs[key] = surgical_data["results"]


# List of (sentence_id, condition, model_spec, region, surprisal)
graph_data = {}

for model_key, surp_df in surp_dfs.items():
    # Collect surprisal data for this model
    graph_data[model_key] = experiment.collect_sentence_surprisals(surp_df)

graph_data = pd.concat(graph_data, names=["model"])

# Now map regions to cross-condition time-points.
graph_times = experiment.plot_config["times"]
region_to_time_idx = {region: time_idx
                      for time_idx, time_regions in enumerate(graph_times.values())
                      for region in time_regions}

graph_data["time"] = graph_data.region.transform(lambda x: region_to_time_idx[x])
graph_data.to_csv("graph_data.csv")

g = sns.FacetGrid(data=graph_data.reset_index(), col="condition", height=10)
g.map(sns.lineplot, "time", "agg_surprisal", "model").add_legend()
for ax in g.axes.ravel():
    ax.set_xticklabels(graph_times.keys())

g.savefig("plot.png")
