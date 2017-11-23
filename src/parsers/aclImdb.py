import os

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
                        content = content_temp
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
    def __init__(self, dataset_dir, wordvec_dir, mode='train', partial_dataset=False):
        self.sentences = SentenceLoader(dataset_dir, with_label=True, full_feature=True, partial_dataset=partial_dataset)
        pass

    # returns [x, y]: feature and label
    def __iter__(self):
        try:
            while True:
                yield self.sentences.next()
        except:
            pass



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
