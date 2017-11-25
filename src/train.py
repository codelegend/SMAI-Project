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
    load the dataset
    '''
    parser = import_module('src.parsers.%s' % args.parser)
    data_loader = parser.DataLoader(
        dataset_dir='datasets/%s' % args.dataset,
        wordvec_dir='var/wordvec/%s/' % args.dataset,
        partial_dataset=False,
        sentence_len=20)

    '''
    Load the model, and setup cuda, if needed
    '''
    model_src = import_module('src.models.%s' % args.model)
    convnet = model_src.Model(sentence_len=data_loader.sentence_len)

    '''
    Train the model
    '''
    train_model(convnet=convnet,
                data_loader=data_loader,
                epochs=100, use_cuda=args.cuda,
                batch_size=100)

'''
Trains the CNN, with multiple epochs

'''
def train_model(convnet, data_loader, epochs=100,
                batch_size=100, shuffle=False, use_cuda=False):
    # Loss function and optimizer for the learning
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(convnet.parameters(), lr=0.001)
    if use_cuda:
        convnet = convnet.cuda()
        loss = loss.cuda()

    num_batches = 1 + (len(data_loader) - 1) / batch_size

    for epoch in xrange(epochs):
        start_time = time.time()

        epoch_loss = 0

        data_iter = iter(data_loader)

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

            # make the batch feature variable
            batch_X = torch.FloatTensor(batch_X)
            if use_cuda: batch_X = batch_X.cuda()
            batch_X = autograd.Variable(batch_X)

            # forward pass
            optimizer.zero_grad()
            output = convnet.forward(batch_X)
            if use_cuda: output = output.cuda()

            # compute loss, and backward pass
            batch_Y = torch.FloatTensor(batch_Y)
            if use_cuda: batch_Y = batch_Y.cuda()
            batch_Y = autograd.Variable(batch_Y).long()
            loss = loss_func(output, batch_Y)
            if use_cuda: loss.cuda()
            loss.backward()
            optimizer.step()

            # update total loss
            epoch_loss += loss.data[0]

            # logging
            if batch_id % 10 == 1:
                logging.debug('Batch %d: loss = %f', batch_id, epoch_loss / batch_id)

            # cleanup
            del batch_X, batch_Y, output, loss



        end_time = time.time()

        ### log epoch execution statistics
        logging.info('Epoch %d: time = %.3f', epoch, end_time - start_time)
        logging.info('> Loss = %f', epoch_loss / num_batches)
