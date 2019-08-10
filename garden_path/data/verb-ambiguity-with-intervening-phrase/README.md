This directory consists of slightly-modified experimental stimuli extracted from various psycholinguistic experiments.

Each CSV row has the following columns:

- Start
- Noun modifier
- Noun
- Unreduced content
- Ambiguous verb
- Unambiguous verb
- RC contents
- RC extra contents pre by-phrase
- RC by-phrase
- RC extra contents post by-phrase
- Disambiguator
- End

## Data sources

Modifications common to all data sources:

- added optional noun modifiers
- distinguished between relative clause contents which are by-phrases and other RC contents

---

- `futrell-modified` is a modified form of [the MV/RC ambiguity stimuli from Futrell et al. (2018): *RNNs as psycholinguistic subjects*][1]. The original data does not differentiate between *by*-phrases and other RC contents; we added by-phrases to this data.
- `clifton2003` is a highly selective subset of the stimuli from [Clifton et al. (2003): *The use of thematic role information in parsing.*][2] We retained only the most plausible stimuli, and used a more diverse set of disambiguating verbs than those in the original stimuli (originally dominated by "was" and "were").
- `levy2009` is a highly selective subset of the stimuli from [Levy et al. (2009): Eye movement evidence that readers maintain and act on uncertainty about past linguistic input.][3]

[1]: https://github.com/Futrell/rnn_psycholinguistic_subjects/tree/master/gardenpathing-verb-ambiguity
[2]: https://www.sciencedirect.com/science/article/pii/S0749596X03000706?via%3Dihub#aep-section-id46
[3]: https://www.pnas.org/content/106/50/21086.short
