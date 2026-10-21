# Data

The dataset is **not included** in this repository.

It was assigned to our group (Group 32) by the instructor of the Data Mining course at the
University of Tehran, and the course rules require working only with that dataset.
It has 100,000 rows, 34 anonymised features (`f1`-`f34`) and a 4-class target `y`
(traffic severity, classes 1-4).

To re-run the notebooks, place the training file here:

```
data/raw/group_32_train.csv
```

The notebooks then write the train/validation/test splits to `data/raw_split/`, the processed
features to `data/processed/`, and the trained models to `models/`. These folders are
git-ignored because of their size.
