'''
Loads sentences from the dataset
Used to learn the word vectors
Handles preprocesing/filtering
'''
# **DO NOT CHANGE THE CLASS NAME**
class SentenceLoader(object):
    def __init__(self, dataset_dir, with_label=False, full_feature=False, partial_dataset=True, mode='train'):
        '''Args for __iter__
        @with_label: return [feature, label] (as array)
        @full_feature: return full feature
        '''
        pass

    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        yield

'''
Loads sentences, and trained word vectors,
and converts the data into wordvectors, by concatenation.
Returns the full vector, with its label.
'''
# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_dir, mode='train', loop_over=False):
        '''
        Load the wordvectors from @wordvec_dir
        modes: train, validate, test
        '''
        # self.sentences = SentenceLoader(dataset_dir, with_label=True, full_feature=True)
        pass

    # returns [x, y]: feature and label
    def __iter__(self):
        yield


helpstr = '''(Version 1.0)
Parser for <dataset>
[Give info here.]
'''
