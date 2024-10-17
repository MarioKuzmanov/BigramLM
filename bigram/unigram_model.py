import numpy as np
from collections import defaultdict

from bigram.constants import *


class Unigram:

    def __init__(self, frequency_threshold):
        self.frequency_threshold = frequency_threshold
        self.training_data = None
        self.test_data = None
        self.vocab = None
        self.counts = None
        self.probs = None

    def preprocess_sentence(self, sent, train=True):
        # if word is not in vocab, we replace with the UNK token
        arr_sent = [w.lower().strip() for w in sent.split(' ') if w not in '\n\t']
        if not train:
            assert self.vocab is not None
            for i in range(len(arr_sent)):
                if arr_sent[i] not in self.vocab:
                    arr_sent[i] = UNK

        return arr_sent

    def load_and_preprocess(self, corpus_file, train=True):
        # data is loaded and preprocessed accordingly
        with open(corpus_file, encoding='utf8') as file:
            arr_sentences = file.readlines()
        arr_data = [self.preprocess_sentence(sent, train) for sent in arr_sentences]
        if train:
            self.training_data = arr_data
        else:
            self.test_data = arr_data

    def build_matrix(self):
        # adds to the count of every token
        self.counts = defaultdict(int)
        for sent in self.training_data:
            for unigram in sent:
                self.counts[unigram] += 1

        # remove low frequency words
        counts_updated = defaultdict(int)
        for key in self.counts:
            if self.counts[key] <= self.frequency_threshold:
                count = self.counts[key]
                counts_updated[UNK] += count
            else:
                counts_updated[key] = self.counts[key]

        self.counts = counts_updated
        self.vocab = sorted(list(self.counts.keys()))

    def calc_probs(self):
        # avoid underflow
        self.probs = self.counts.copy()
        sum1 = sum(self.counts.values())
        for w in self.vocab:
            self.probs[w] = np.log10(self.probs[w] / sum1)

    def train(self, train_file):
        self.load_and_preprocess(corpus_file=train_file, train=True)
        self.build_matrix()
        self.calc_probs()

    def perplexity(self, test_sentences):
        logprobs2, N = 0, 0
        for sent in test_sentences:
            N += len(sent)
            logprobs2 += sum([self.probs[word] for word in sent])
        return 10 ** ((-logprobs2) / N)

    def test(self, test_file):
        self.load_and_preprocess(corpus_file=test_file, train=False)
        pp = self.perplexity(test_sentences=self.test_data)
        return pp


if __name__ == '__main__':
    unigram_model = Unigram(frequency_threshold=1)
    unigram_model.train(train_file="../data/SherlockHolmes-train.txt")
    pp = unigram_model.test(test_file="../data/SherlockHolmes-test.txt")
    print("Perplexity: {}".format(round(pp, 3)))
