import sys, time, os, logging
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from importlib import import_module
from sklearn import metrics

'''
Testing:
Load trained weights from file, and feed them to the model.
Test the data, and report scores.
'''
def test(args): # DO NOT EDIT THIS LINE
    '''
    load the dataset
    '''
    parser = import_module('src.parsers.%s' % args.parser)
    data_loader = parser.DataLoader(
        dataset_dir='datasets/%s' % args.dataset,
        wordvec_dir='var/wordvec/%s/' % args.dataset,
        partial_dataset=False,
        sentence_len=20,
        mode='test',
        shuffle=True)

    '''
    Load the network and the saved weights
    '''
    model_src = import_module('src.models.%s' % args.model)
    convnet = model_src.Model(sentence_len=data_loader.sentence_len)

    weights_dir = 'var/train/%s.%s' % (args.model, args.dataset)
    if args.load_from is None:
        args.load_from = os.listdir(weights_dir)[0]
    weights_file = '%s/%s.pt' % (weights_dir, args.load_from)
    logging.info('Loading weights from %s', weights_file)
    state_checkpoint = torch.load(weights_file)
    convnet.load_state_dict(state_checkpoint)

    '''
    pass the data through the network
    '''
    run_tests(convnet=convnet, data_loader=data_loader,
              use_cuda=args.cuda)

'''
Run the tests on the test samples
'''
def run_tests(convnet, data_loader,
              batch_size=100, use_cuda=False):

    data_iter = iter(data_loader)
    num_batches = 1 + (len(data_loader) - 1) / batch_size

    actual, predicted = np.array([]), np.array([])
    for batch_id in xrange(num_batches):
        # load current batch
        batch_X, batch_Y = [], []
        try:
            for i in xrange(batch_size):
                feature, label = data_iter.next()
                batch_X.append(feature)
                batch_Y.append(label)
        except StopIteration:
            pass
        batch_X = torch.FloatTensor(batch_X)
        if use_cuda: batch_X = batch_X.cuda()
        batch_X = autograd.Variable(batch_X)
        output = convnet.forward(batch_X)

        _, pred = output.max(1)
        predicted = np.append(predicted, [pred.data.numpy()])
        actual = np.append(actual, [batch_Y])

        if (batch_id + 1) % 10 == 0:
            logging.debug('Batch %d done', batch_id+1)

    ### Compute scores
    logging.info('Prediction done. Scores:')
    logging.info('> accuracy = %f', metrics.accuracy_score(actual, predicted))
    logging.info('> precision = %f', metrics.precision_score(actual, predicted))
    logging.info('> recall = %f', metrics.recall_score(actual, predicted))
    logging.info('> F1 = %f', metrics.f1_score(actual, predicted))
