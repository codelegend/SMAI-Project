#!/usr/bin/python
import os, sys, logging
import argparse
from importlib import import_module

def check_directory_structure():
    dirs = ['var', 'var/train', 'var/wordvec', 'var/log', 'var/log/train', 'var/log/test', 'datasets']
    for d in dirs:
        if not os.path.isdir(d):
            logging.critical("Directory `%s` not found" % d)
            return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Sentence Classification')
    parser.add_argument('task', metavar='TASK',
        choices=['preprocess', 'train', 'test'],
        help='Task to perform: {preprocess, train, test}')

    parser.add_argument('--model', dest='model',
        default='model_example',
        help='Model to use (src/models/MODEL.py)')
    parser.add_argument('--dataset', dest='dataset',
        default=None,
        help='Dataset to use (dataset/DATASET/)')
    parser.add_argument('--parser', dest='parser',
        default='parser_example',
        help='Data parser to use (stored in src/parsers/PARSER.py)')

    parser.add_argument('--load-from', dest='load_from',
        default=None,
        help='Information about backup to load from.')
    parser.add_argument('--output', dest='output',
        default=None,
        help='Output folder/file')

    parser.add_argument('--log-level', dest='log_level',
        type=int, default=30,
        help='Logging level (10 for testing, and 40 for production. Default: 30)')

    parser.add_argument('--gpu', dest='cuda',
        action='store_true',
        help='Run code on GPU (uses cuda)')

    return parser.parse_args()

def main():
    # logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=0)
    logging.basicConfig(format='[%(levelname)s] %(filename)s #%(lineno)d %(message)s', level=0)

    args = parse_args()
    if args.log_level:
        logging.basicConfig(level=args.log_level)
        del args.log_level

    if not check_directory_structure():
        sys.exit(1)

    if args.cuda:
        import torch
        args.cuda = torch.cuda.is_available()
        logging.info("CUDA Available: %s", 'YES' if args.cuda else 'NO')

        # some cuda config
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True


    if args.task == 'preprocess':
        import_module('src.preprocess') \
        .learn_word_vectors(dataset=args.dataset,
                            parser_name=args.parser,
                            output_dir='var/wordvec/%s' % args.dataset)
    elif args.task == 'train':
        import_module('src.train').train(args)
    elif args.task == 'test':
        import_module('src.test').test(args)

    logging.info('Exiting...')

main()
