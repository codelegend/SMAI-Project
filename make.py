#!/usr/bin/python
import os, sys, logging, logging.config, yaml
import argparse
from importlib import import_module

def setup_logging(args):
    config = {}
    with open('logging.yaml') as f:
        config = yaml.safe_load(f)
        log_file = 'var/log/%s/' % args.task
        if args.task == 'preprocess':
            log_file += args.parser
        else:
            log_file += args.model
        log_file += '.log'
        config['handlers']['file_handler']['filename'] = log_file
    logging.config.dictConfig(config)


def check_directory_structure():
    dirs = ['var', 'var/train', 'var/wordvec', 'var/log', 'var/log/train', 'var/log/test', 'var/log/preprocess', 'datasets']
    for d in dirs:
        if not os.path.isdir(d):
            logging.warn("Directory `%s` not found, creating." % d)
            os.mkdir(d)
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
        help='Output file')

    parser.add_argument('--gpu', dest='cuda',
        action='store_true',
        help='Run code on GPU (uses cuda)')

    return parser.parse_args()

def main():
    if not check_directory_structure():
        sys.exit(1)

    args = parse_args()
    setup_logging(args)

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

    logging.info('Exiting...\n%s', '-' * 80)

main()
