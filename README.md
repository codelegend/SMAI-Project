SMAI Project: Sentence Classification using deep learning
==========================================================

Directory Structure:
--------------------
```
.
├── datasets/
│   ├── <dataset_name>/
│   └── ...
├── doc/
├── src/
│   ├── models/
│   │   ├── <model_name.py>
│   │   └── ...
│   ├── parsers/
│   │   ├── <parser_name.py>
│   │   └── ...
│   ├── train.py
│   └── test.py
└── var/ (not committed)
    ├── train/
    │   └── <model_name>/
    │       └── <dataset_name>/
    └── wordvec/
        └── <dataset_name>/
```

Usage:
------
`make.py [-h] [--model MODEL] [--dataset DATASET] [--parser PARSER]
               [--log-level LOG_LEVEL]
               TASK`
- `TASK`: Task to perform `{preprocess, train, test}`
- `--model MODEL`: Model to use (`src/models/MODEL.py`)
- `--dataset DATASET`: Dataset to use (`dataset/DATASET/`)
- `--parser PARSER`: Data parser to use (`src/parsers/PARSER.py`)
- `--log-level LOG_LEVEL`: Logging level (10 for testing, and 40 for production. Default: 30)

Adding Models:
--------------
Check Readme in `src/models/`

Adding Parsers:
---------------
Adding a parser for new datasets - to handle preprocessing and conversion to word vectors.
Check Readme in `src/parsers/`

Adding Datasets:
----------------
Add datasets to the `datasets` folder. Parsers should know the corresponding internal directory structure.

## Contributors
Team: `\sigma \sqrt -1`
- Arjun P
- Gokul B Nair
- Soumya Vadlamannati
- Sai Anurudh Reddy Peduri
