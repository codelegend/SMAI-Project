#!/usr/bin/python
import os, sys, argparse, logging


def check_directory_structure():
    dirs = ['var', 'datasets2']
    for d in dirs:
        if not os.path.isdir(d):
            logging.error("Directory %s not found" % d)
            return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Sentence Classification')
    parser.add_argument('task', metavar='TASK',
        choices=['preprocess', 'train', 'test'],
        help='Task to perform: {preprocess, train, test}')

    parser.add_argument('--model', dest='model',
        default='default',
        help='Model to use (src/models/MODEL.py)')
    parser.add_argument('--dataset', dest='dataset',
        default='default',
        help='Dataset to use (dataset/DATASET/)')
    parser.add_argument('--parser', dest='parser',
        default='default',
        help='Data parser to use (stored in src/parsers/PARSER.py)')

    parser.add_argument('--log-level', dest='log_level',
        type=int, default=30, help='Logging level (10 for testing, and 40 for production. Default: 30)')

    return parser.parse_args()

def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] {%(filename)s #%(lineno)d} %(message)s',
        level=0)
    args = parse_args()
    if args.log_level:
        logging.basicConfig(level=args.log_level)
        del args.log_level
    if not check_directory_structure():
        sys.exit(1)

    # if args.preprocess:

main()
