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
        try:
            with open(fname) as f:
                for line in open(os.path.join(dirname, fname)):
                    sanitised_line = re.sub('<br />|<i/?>|<hr>', '', line)
                    for sentence in re.split('[.!?]', sanitised_line):
                        concat = []
                        for word in re.split('[,; ]', sentence):
                            concat.append(model[word])
                        concat.append([0]*(WORDS_IN_SENTENCE*100 - length(concat)))
                        examples.append(concat)
        except:
            print(fname,"File can't be opened")
    return examples

def load_imdb(file_prefix):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec.load('../models/word2vec')

    examples = []
    labels = []
    pos_examples = load_class(model, file_prefix + 'pos')
    examples.append(pos_examples)
    labels.append([1]*(length(pos_examples)/100))
    neg_examples = load_class(model, file_prefix + 'neg')
    labels.append([0]*(length(neg_examples)/100))
    examples.append(neg_examples)
    return (examples, labels)

def train(x, y, epochs=10, batch_size=50):
    net = model.SentenceNet()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.SDG(net.parameters(), lr=0.001)
    data_size = len(x)
    num_batches = int(data_size/batch_size)
    count = 0
    for epoch in range(epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y

        final_loss = 0
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch = x_batch[start_index:end_index]
            y_batch = y_batch[start_index:end_index]
            batch = torch.FloatTensor(x_batch).view(len(x_batch), 100, 100)
            batch = autograd.Variable(batch)
            optimizer.zero_grad()
            out = net.forward(batch)
            _, pred = out.max(1)
            target = y_batch[:,1]
            target = np.float32(target)
            target = autograd.Variable(torch.FloatTensor(target)).long()
            output = loss(out, target)
            output.backward()
            optimizer.step()
            count += 1
            final_loss += output.data[0]
            if batch_num % 100 == 0:
                print("Batch Number: " + str(batch_num))
                print("Loss: ", final_loss/(batch_num+1))
        print("Loss after epoch " + str(epoch) + " = " + str(final_loss/num_batches))
        # f = str('train_backup' + str(epoch) + '.pt')
        # torch.save(net.state_dict(), f)

(X, Y) = load_imdb('../aclImdb/train/')
train(X, Y)
