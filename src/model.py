import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SentenceNet(nn.Module):
    def __init__(self, wordvec_len=100, num_filters=10, sentence_len=100):
        super(SentenceNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 100, (3, 128))
        self.pool1 = nn.MaxPool1d(1)
        self.fc1 = nn.Linear(100, 100)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x

    def num_flat_features(self, x):
        return reduce(lambda a, b: a * b, x.size()[1:])

# n = SentenceNet()
# print n
