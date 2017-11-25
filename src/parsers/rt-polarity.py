import os, copy
import gensim
import numpy as np
import re

# **DO NOT CHANGE THE CLASS NAME**
class SentenceLoader(object):
    def __init__(self, dataset_dir, with_label=False, full_feature=False, partial_dataset=False, mode='train', shuffle=False):
        '''Args for __iter__
        @with_label: return [feature, label] (as array)
        @full_feature: return full feature
        '''
        self.with_label = with_label
        self.full_feature = full_feature

        # test set is labelled
        if mode == 'validate': mode = 'test'

        self.dataset_files = []
        # load pos samples
        self.dataset_files.append((os.path.join(dataset_dir,mode,'rt-polarity.neg'), 0))
        self.dataset_files.append((os.path.join(dataset_dir,mode,'rt-polarity.pos'), 1))

        # truncate if partial_dataset
        # if partial_dataset:
        #     self.dataset_files = self.dataset_files[:10]

        # config
        self.shuffle = shuffle


    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        dataset = copy.copy(self.dataset_files)
        if self.shuffle:
            np.random.shuffle(dataset)

        for fname, label in dataset:
            with open(fname) as f:
                full_text = f.read().strip()
                reviews = full_text.split('\n')
                for content in reviews:
                    content = self.process_line(content)
                    if self.full_feature:
                        content_temp = []
                        for c in content: content_temp.extend(c)
                        content = [content_temp]
                    if self.with_label:
                        content = map(lambda line: (line, label), content)

                    for line in content:
                        yield line

    def __len__(self):
        return len(self.dataset_files)

    # return the lines after splitting into words, and filtering.
    def process_line(self, content):
        content = re.sub(r"[^A-Za-z0-9,.!?\']"," ",content)
        content = re.sub(r"n't"," not ",content)
        content = re.sub(r"'nt"," not ",content)
        content = re.sub(r"'","",content)
        content = re.sub(r","," , ",content)
        content = re.sub(r"!"," ! ",content)
        content = re.sub(r"\?"," ? ",content)
        content = re.sub(r". . .","",content)
        content = re.sub(r"- -","",content)
        content = re.sub(r"Mr.","Mr",content)
        content = re.sub(r"Mrs.","Mrs",content)
        content = re.sub(r"Ms.","Ms",content)
        content = content.split('.')
        content = [line.split() for line in content]
        return content

# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_dir, wordvec_file=None,
                mode='train', partial_dataset=False, shuffle=False,
                sentence_len=10, wordvec_dim=100):
        # load sentences
        self.sentences = SentenceLoader(dataset_dir,
                                        with_label=True,
                                        full_feature=True,
                                        partial_dataset=partial_dataset, shuffle=shuffle)

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
        self.sentence_len = sentence_len
        self.wordvec_dim = wordvec_dim
        self.partial_dataset = partial_dataset
        self.dataset_dir = dataset_dir

    # returns [x, y]: feature and label
    def __iter__(self):
        for sentence, label in self.sentences:
            sentence.reverse()
            wordvec = np.ndarray((0, self.wordvec_dim))
            count = 0 # only add `self.sentence_len` words
            for word in sentence:
                if word in self.word_vectors.vocab:
                    wordvec = np.append(wordvec, [self.word_vectors[word]], axis=0)
                    count += 1
                    if count == self.sentence_len:
                        break

            # pad with zeros, if sentence is too small
            if count < self.sentence_len:
                wordvec = np.append(wordvec, np.zeros((self.sentence_len - count, self.wordvec_dim)), axis=0)
            yield wordvec, label

    def __len__(self):
        return len(self.sentences)

helpstr = '''(Version 1.0)
Parser for rt Dataset
Directory structure:
<root>
    -train
        - pos_file
        - neg_file
    -test
        - pos_file
        - neg_file
'''
