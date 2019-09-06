"""
Shared experimental utilities.
"""

from pathlib import Path

import pandas as pd
import yaml


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
        for condition in self.conditions:
            self._check_region_refs(condition["prefix_columns"])
            self._check_region_refs([condition["extract_column"]])
            self._check_region_refs([condition["measure_column"]])
        self._check_condition_refs(self.decoder_config["train_conditions"])
        for plot_time, regions in self.plot_config["times"].items():
            self._check_region_refs(regions)

    @classmethod
    def from_yaml(self, yaml_path):
        return cls(yaml.load(yaml_path), yaml_path.parent)

    def _load_stimuli(self, path):
        return pd.read_csv(path, header=0, index_col=0).sort_index()

    def _check_condition_refs(self, condition_list):
        assert set(condition_list).issubset(set(self.conditions.keys())), \
            ("Provided condition list (%s) is a superset of conditions in experiment (%s)"
             % (",".join(condition_list), ",".join(self.conditions.keys())))

    def _check_region_refs(self, region_list):
        assert set(region_list).issubset(self.stimulus_regions), \
            ("Provided region list (%s) is a superset of regions in input stimuli (%s)"
             % (",".join(region_list), ",".join(self.stimulus_regions)))

    def get_sentences(self, yield_extract_idxs=False):
        """
        Generate individual sentence stimuli by crossing items with conditions.

        Args:
            yield_extract_idxs: If `True`, yield tuples `(sentence_str,
              extract_idx)`, where `extract_idx` is the zero-based token index
              of the final token of the extract column for this sentence.
        """
        for _, row in self.stimuli.iterrows():
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

                if yield_extract_idxs:
                    yield sentence, extract_column_token_idx
                else:
                    yield sentence

    def collect_sentence_surprisals(self, surprisal_df):
        """
        Given a surprisal dataframe, link these surprisals to individual
        sentences and items.
        """
        for idx, sentence in enumerate(self.get_sentences()):
            sentence_surprisals = surprisal_df.loc[idx + 1]
            sentence_tokens = sentence.split(" ")


def read_surprisal_df(path):
    """
    Read a language model surprisal output dataframe.
    """
    return pd.read_csv(path, header=0, index_col=["sentence_id", "token_id"], sep="\t") \
            .sort_index()
