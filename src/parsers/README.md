Data Parsing/Preprocessing
==========================
Modules/functions for preprocessing data, and converting them to word vectors.  
Check `parser_example.py` for more details.

Modules
-------
- `class Word2Vec`
  - Handles word vector generation
  - Converts data to word vectors, and stores it in `var/wordvec/<dataset>/`


- `class DataLoader`
  - Loads the data/text from the dataset (`datasets/<dataset>/`)
  - Must support batch loading.
