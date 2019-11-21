"""
Shared experimental utilities.
"""

from collections import namedtuple
from pathlib import Path

import pandas as pd
import yaml


Sentence = namedtuple("Sentence", ["text", "item_idx", "condition", "extract_idx"])

class Experiment(object):

    def __init__(self, yaml_spec, path_root=None):
        e = yaml_spec["experiment"]

        self.name = e["name"]
        self.conditions = e["conditions"]
        self.decoder_config = e["decoder"]
        self.surgery_config = e["surgery"]
        self.plot_config = e["plot"]

        path_root = path_root or Path(__file__).absolute()
        self.stimuli_path = path_root / e["stimuli"]
        self.stimuli = self._load_stimuli(self.stimuli_path)
        self.stimulus_regions = list(self.stimuli.columns)

        # Run validation checks.
        for condition in self.conditions.values():
            self._check_region_refs(condition["prefix_columns"])
            self._check_region_refs([condition["extract_column"]])
            self._check_region_refs([condition["measure_column"]])
        self._check_condition_refs(self.decoder_config["train_conditions"])
        for plot_time, regions in self.plot_config["times"].items():
            self._check_region_refs(regions)

    @classmethod
    def from_yaml(cls, yaml_path):
        with yaml_path.open("r") as yaml_f:
            yaml_data = yaml.load(yaml_f, Loader=yaml.SafeLoader)
        return cls(yaml_data, yaml_path.parent)

    def _load_stimuli(self, path):
        ret = pd.read_csv(path, header=0).fillna("")
        if "source" in ret.columns:
            index = ["source", "idx"]
        else:
            index = ["idx"]

        return ret.set_index(index).sort_index()

    def _check_condition_refs(self, condition_list):
        assert set(condition_list).issubset(set(self.conditions.keys())), \
            ("Provided condition list (%s) is a superset of conditions in experiment (%s)"
             % (",".join(condition_list), ",".join(self.conditions.keys())))

    def _check_region_refs(self, region_list):
        assert set(region_list).issubset(self.stimulus_regions), \
            ("Provided region list (%s) is a superset of regions in input stimuli (%s)"
             % (",".join(region_list), ",".join(self.stimulus_regions)))

    def get_sentences(self):
        """
        Generate individual Sentence objects by crossing items with conditions.
        """
        for item_idx, row in self.stimuli.iterrows():
            for name, condition in self.conditions.items():
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

                yield Sentence(text=sentence, condition=name,
                               item_idx=item_idx,
                               extract_idx=extract_column_token_idx)

    def collect_sentence_surprisals(self, surprisal_df, agg=sum):
        """
        Join stimuli with computed surprisals, aggregating surprisal over
        stimulus regions.

        Args:
            agg: Aggregation function to apply to the sequence of tokens in
                each region of each sentence
        """
        ret_df = []
        surprisal_df = surprisal_df.reset_index().set_index(["sentence_id", "token_id"])
        for idx, sentence in enumerate(self.get_sentences()):
            sentence_surprisals = surprisal_df.loc[idx + 1]
            sentence_tokens = sentence.text.split(" ")

            item = self.stimuli.loc[sentence.item_idx]
            i = 0
            for region in self.conditions[sentence.condition]["prefix_columns"]:
                region_tokens = item[region].strip().split(" ")
                region_surprisals = sentence_surprisals.iloc[i:i + len(region_tokens)].surprisal
                assert len(region_tokens) == len(region_surprisals)

                ret_df.append((idx, sentence.item_idx, sentence.condition, region, agg(region_surprisals)))
                i += len(region_tokens)

        return pd.DataFrame(ret_df, columns=["index", "item_idx", "condition", "region", "agg_surprisal"]) \
                .set_index("index")


def read_surprisal_df(path):
    """
    Read a language model surprisal output dataframe.
    """
    return pd.read_csv(path, header=0, index_col=["sentence_id", "token_id"], sep="\t") \
            .sort_index()
