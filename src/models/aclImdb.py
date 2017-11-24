from torch import nn
import numpy as np

# **DO NOT CHANGE THE CLASS NAME**
class Model(nn.Module):
    def __init__(self, wordvec_dim=100, filter_sizes=[(3, 100), (4, 100), (5, 100)]):
        super(Model, self).__init__()
        '''
        Add layers here
        '''

        # Convolution layer
        self.convs = []
        for filter_width, num_filters in filter_sizes:
            self.convs.append(nn.Conv2d(1, num_filters, (filter_width, wordvec_dim))
        self.convs = nn.ModuleList(self.convs)

        # max pooling

        # NN


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
