#!/bin/env python

import gensim, logging, os, re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SentenceIterator(object):
    def __init__(self, dir_prefix):
        self.dir_prefix = dir_prefix

    def __iter__(self):
      for suffix in ['test/pos', 'test/neg', 'train/pos', 'train/neg']:
        dirname = self.dir_prefix + suffix
        for fname in os.listdir(dirname):
            for line in open(os.path.join(dirname, fname)):
                yield re.split('[.!?] ', line) # utf-8 encoding needed?

sentences = SentenceIterator('../datasets/')
model = gensim.models.Word2Vec(sentences, min_count=3)
model.save('../models/word2vec')