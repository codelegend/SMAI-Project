# TODO: everything...
import sys, os, logging
from importlib import import_module
import gensim
workers = 4 ### number of workers for gensim

'''
Preprocessing:
@arg dataset Load from `datasets/@dataset`
@arg parser_name Use parser src/parsers/@parser_name
@arg output_dir Save word vectors to `var/wordvec/@output_dir`
'''
def learn_word_vectors(dataset, parser_name, output_dir): # DO NOT EDIT THIS LINE
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    parser = import_module('src.parsers.%s' % parser_name)
    sentences = parser.SentenceLoader(dataset_dir='datasets/%s' % dataset, partial_dataset=False)

    model = gensim.models.Word2Vec(sentences, min_count=5, size=100, workers=workers)
    # for key in model.wv.vocab:
    #     print key

    save_file = 'model_%d.gensim' % get_next_unused_id(output_dir)
    model.save(os.path.join(output_dir, save_file))

def get_next_unused_id(output_dir):
    files = os.listdir(output_dir)
    files = map(lambda f: int(f.replace('model_', '').replace('.gensim', '')), files)
    files.sort()
    file_id = 1
    for i in files:
        if file_id == i:
            file_id += 1
    return file_id
