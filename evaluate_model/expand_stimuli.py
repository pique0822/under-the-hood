from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def main(args):
    data = pd.read_csv(args.stimuli_file).fillna("")

    verbs = pd.DataFrame([])
    if args.extra_verbs is not None:
        verbs = pd.read_csv(args.corpus_stats_file)

    expanded = []
    for _, row in data.iterrows():
        for ambiguous in [True, False]:
            for reduced in [True, False]:
                sentence = row.Start.split() + row.Noun.split()
                if not reduced:
                    sentence.extend(row["Unreduced content"].split())

                if ambiguous:
                    sentence.append(row["Ambiguous verb"])
                else:
                    sentence.append(row["Unambiguous verb"])

                if row["RC contents"]:
                    sentence.extend(row["RC contents"].split())
                else:
                    sentence.extend(row["RC by-phrase"].split())
                # sentence.append(row.Intervener)

                # Build data item with original verb.
                expanded.append({
                    "sentence": " ".join(sentence + [row.Disambiguator, "<eos>"]),
                    "ambiguous": ambiguous,
                    "reduced": reduced,
                    "disambiguator_idx": len(sentence),
                })

                for _, verb_row in verbs.iterrows():
                    expanded.append({
                        "sentence": " ".join(sentence + [verb_row.token.lower(), "<eos>"]),
                        "ambiguous": ambiguous,
                        "reduced": reduced,
                        "disambiguator_idx": len(sentence),
                    })

    df = pd.DataFrame.from_records(expanded)
    with args.out_file.open("w") as out_f:
        df.to_csv(out_f, index=False)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("stimuli_file", type=Path)
    p.add_argument("out_file", type=Path)
    p.add_argument("--extra_verbs", type=Path)

    args = p.parse_args()
    main(args)
