import os
import gensim
import numpy as np
import logging

# **DO NOT CHANGE THE CLASS NAME**
class SentenceLoader(object):
    def __init__(self, dataset_dir, with_label=False, full_feature=False, partial_dataset=False, mode='train'):
        '''Args for __iter__
        @with_label: return [feature, label] (as array)
        @full_feature: return full feature
        '''
        self.with_label = with_label
        self.full_feature = full_feature

        # test set is labelled
        if mode == 'validate': mode = 'test'

        # load pos samples
        self.pos_dir = os.path.join(dataset_dir, mode, 'pos')
        self.pos_files = os.listdir(self.pos_dir)

        # load neg samples
        self.neg_dir = os.path.join(dataset_dir, mode, 'neg')
        self.neg_files = os.listdir(self.neg_dir)

        # truncate if partial_dataset
        if partial_dataset:
            self.pos_files = self.pos_files[:10]
            self.neg_files = self.neg_files[:10]


    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        dataset = [(self.pos_files, self.pos_dir, 1),
            (self.neg_files, self.neg_dir, 0)]

        for file_des in dataset:
            # file_des := (file_list, directory, label)
            for fname in file_des[0]:
                with open(os.path.join(file_des[1], fname)) as f:
                    content = f.read().strip()
                    content = self.process_line(content)
                    if self.full_feature:
                        content_temp = []
                        for c in content: content_temp.extend(c)
                        content = [content_temp]
                    if self.with_label:
                        content = map(lambda line: (line, file_des[2]), content)

                    for line in content:
                        yield line

    # return the lines after splitting into words, and filtering.
    def process_line(self, content):
        content = content.replace('Mr.', 'Mr')
        content = content.replace('Mrs.', 'Mrs')
        content = content.replace('Ms.', 'Ms')
        content = content.split('.')
        content = [line.split() for line in content]
        return content

# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_dir, mode='train', partial_dataset=False, wordvec_file=None, num_words=10, wordvec_dim=100):
        # load sentences
        self.sentences = SentenceLoader(dataset_dir, with_label=True, full_feature=True, partial_dataset=partial_dataset)

        # load the word vectors
        if wordvec_file is None:
            files = os.listdir(wordvec_dir)
            files.sort()
            wordvec_file = files[-1]
        wordvec_file = os.path.join(wordvec_dir, wordvec_file)
        model = gensim.models.Word2Vec.load(wordvec_file)
        self.word_vectors = model.wv # no updates
        del model

        # config
        self.num_words = num_words
        self.wordvec_dim = wordvec_dim
        self.partial_dataset = partial_dataset
        self.dataset_dir = dataset_dir

    # returns [x, y]: feature and label
    def __iter__(self):
        for sentence, label in self.sentences:
            sentence.reverse()
            wordvec = np.ndarray((0, self.wordvec_dim))
            count = 0 # only add `self.num_words` words
            for word in sentence:
                if word in self.word_vectors.vocab:
                    wordvec = np.append(wordvec, [self.word_vectors[word]], axis=0)
                    count += 1
                    if count == self.num_words:
                        break

            # pad with zeros, if sentence is too small
            if count < self.num_words:
                wordvec = np.append(wordvec, np.zeros((self.num_words - count, self.wordvec_dim)), axis=0)
            yield wordvec, label

helpstr = '''(Version 1.0)
Parser for aclImdb Dataset
Directory structure:
<root>
    - train
        - pos
        - neg
    - test
        - pos
        - neg
        - unsup
'''
