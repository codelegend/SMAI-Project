import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# **DO NOT CHANGE THE CLASS NAME**
class Model(nn.Module):
    def __init__(self, num_classes=2,
        sentence_len=100, wordvec_dim=100,
        filter_sizes=[(3, 100), (4, 100), (5, 100)]):

        super(Model, self).__init__()
        self.num_classes = num_classes
        
        # Convolution layer and max pooling
        self.convs = []
        self.maxpools = []
        fc_size = 0 # input size to fully connected layer
        for filter_width, num_filters in filter_sizes:
            self.convs.append(
                # (input_channels, output_channels, (filter_width, wordvec_dim))
                nn.Conv2d(1, num_filters, (filter_width, wordvec_dim))
            )
            self.maxpools.append(
                nn.MaxPool1d(sentence_len - filter_width + 1)
            )
            fc_size += num_filters

        self.convs = nn.ModuleList(self.convs)
        self.maxpools = nn.ModuleList(self.maxpools)

        # NN
        layer_sizes = [fc_size, 100]
        self.FC1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = nn.Dropout(p = 0.5)
        self.layer = nn.Linear(layer_sizes[1], 2)

        # random-initialization

    def conv_pool(self, x, conv, pool):
        return x


    def forward(self, x):
        '''
        Pass x through the CNN
        '''
        return x

    def num_flat_features(self, x):
        return reduce(lambda a, b: a * b, x.size()[1:])

helpstr = '''(Version 1.0)
Example model CNN
@input: any Tensor
@returns: @input tensor
'''
