class DataLoader:
    def __init__(self, dataset_dir):
        pass

    # returns [x, y]: feature and label
    def get_next_feature(self):
        pass

    # return a batch of feature vectors
    def get_next_batch(self, batch_size=50):
        pass

class Word2Vec:
    def __init__(self, dataset_dir, output_dir, output_file):
        pass

    # Generates the word vectors and writes to output_dir/output_file
    def process(self, wordveclen=100):
        pass
