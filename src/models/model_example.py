from torch import nn
import numpy as np

# DO NOT CHANGE THE CLASS NAME
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x):
        return x

    def num_flat_features(self, x):
        return reduce(lambda a, b: a * b, x.size()[1:])
