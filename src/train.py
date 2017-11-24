import sys, time, logging
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from importlib import import_module


'''
Training:
- Use the parser class to load the data, and feed it to the model.
- Log training scores and loss (to `var/log/...`)
- Store weights to `var/train/...`
- Checkpoint weights at regular intervals
'''
def train(args): # DO NOT EDIT THIS LINE
    '''
    Load the model, and setup cuda, if needed
    '''
    model_src = import_module('src.models.%s' % args.model)
    convnet = model_src.Model()

    '''
    load the dataset
    '''
    parser = import_module('src.parsers.%s' % args.parser)
    data_loader = parser.DataLoader(
        dataset_dir='datasets/%s' % args.dataset,
        wordvec_dir='var/wordvec/%s/' % args.dataset,
        partial_dataset=True,
        num_words=20)

    '''
    Train the model
    '''
    train_model(convnet=convnet,
                data_loader=data_loader,
                epochs=2, use_cuda=args.cuda)

def train_model(convnet, data_loader, epochs=100,
                batch_size=100, shuffle=False, use_cuda=False):
    # Loss function and optimizer for the learning
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(convnet.parameters(), lr=0.001)
    if use_cuda:
        convnet = convnet.cuda()
        loss = loss.cuda()

    features = []
    # NOTE: each label in labels is a probability distribution
    #       so, if the class is known, then it looks like [1, 0, 0...]
    labels = []
    for feature, label in data_loader:
        features.append(feature)
        label_dist = [0] * convnet.num_classes
        label_dist[label] = 1
        labels.append(label_dist)
    features = np.array(features)
    labels = np.array(features)

    num_batches = 1 + (len(features) - 1) / batch_size

    for epoch in xrange(epochs):
        start_time = time.time()

        '''
        Run an epoch
        Feed the data in batches of `batch_size`
        '''
        train_X, train_Y = features, labels
        train_X = np.array_split(train_X, num_batches)
        train_Y = np.array_split(train_Y, num_batches)

        epoch_loss = 0

        for batch_id in xrange(num_batches):
            # make the batch feature variable
            batch_X = torch.FloatTensor(train_X[batch_id])
            if use_cuda: batch_X = batch_X.cuda()
            batch_X = autograd.Variable(batch_X)

            # forward pass
            optimizer.zero_grad()
            output = convnet.forward(batch_X)
            if use_cuda: output = output.cuda()
            _, pred = output.max(1)

            # compute loss, and backward pass
            batch_Y = train_Y[batch_id]
            batch_Y = torch.FloatTensor(batch_Y)
            if use_cuda: batch_Y = batch_Y.cuda()
            batch_Y = autograd.Variable(batch_Y)

            loss = loss_func(output, batch_Y)
            if use_cuda: loss.cuda()
            loss.backward()
            optimizer.step()

            # update total loss
            epoch_loss += loss.data[0]

            # logging
            if batch_id % 100 == 1:
                logging.debug('Batch %d: loss = %f', batch_id, epoch_loss / batch_id)


        end_time = time.time()

        ### log epoch execution statistics
        logging.info('Epoch %d: time = %.3f', epoch, end_time - start_time)
        logging.info('> Loss = %f', epoch_loss / num_batches)
