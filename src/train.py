import numpy as np
import json
import sys
import model
import torch
from torch import autograd
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle

import gensim, logging, os, re

WORDS_IN_SENTENCE = 100

def load_class(model, dirname):
    examples = []
    for fname in os.listdir(dirname):
        with open(fname) as f:
            for line in open(os.path.join(dirname, fname)):
                sanitised_line = re.sub('<br />|<i/?>|<hr>', '', line
                for sentence in re.split('[.!?] ', sanitised_line
                    concat = []
                    for word in re.split('[,; ]', sentence)
                        concat.append(model[word])
                    concat.append()
                    examples.append(concat)
    return examples

def load_imdb(file_prefix):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec.load('../models/word2vec')

    examples = []
    labels = []
    pos_examples = load_class('pos')
    examples.append(pos_examples)
    neg_examples = load_class('neg')
    examples.append(pos_examples)

def train():
