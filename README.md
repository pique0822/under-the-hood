Current pipeline

1. (TODO: automate) Critical disambiguation-site surprisals are stored in [`surprisal_rc.csv`][1]
2. [`surgical_pipeline.sh`][2] ...
  1. learns a decoder to predict the surprisal data
  2. runs surgery experiments
  3. plots results

[1]: https://github.com/pique0822/under-the-hood/blob/master/evaluate_model/surprisal_rc.csv
[2]: https://github.com/pique0822/under-the-hood/blob/master/surgical_pipeline.sh

#### Setup
```
cd under-the-hood
python setup.py install
```
