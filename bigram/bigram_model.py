from bigram.unigram_model import *
from bigram.constants import *
from tqdm import tqdm
import time


class Bigram():
    def __init__(self, frequency_threshold=1):

        self.frequency_threshold = frequency_threshold
        self.unigram_model = None

        self.training_data = None
        self.test_data = None
        self.vocab = None
        self.row_idx_map, self.col_idx_map = None, None
        self.matrix = None
        self.probs = None

        self.unigrams = None
        self.bt_begin = None
        self.bt_end = None
        self.pp = None

    def preprocess_sentence(self, sent, train=True):
        # marks the words not in vocab with UNK and adds markers
        if not train:
            sent = [w.lower().strip() for w in sent.split(' ') if w not in '\n\t']

        for i in range(len(sent)):
            if sent[i] not in self.vocab:
                sent[i] = UNK
        sent = [BOS] + sent + [EOS]

        return sent

    def load_and_preprocess(self, train=True):
        # build training/test data based on the preprocessed sentences.

        if train:
            self.training_data = [self.preprocess_sentence(sent) for sent in self.unigram_model.training_data]
        else:
            self.test_data = [self.preprocess_sentence(sent) for sent in self.unigram_model.test_data]

        self.vocab = self.unigram_model.vocab

    def set_mappings(self):
        self.row_idx_map = {self.vocab[i]: i for i in range(len(self.vocab))}
        self.row_idx_map.update({BOS: len(self.vocab), BT_END: len(self.vocab) + 1})

        self.col_idx_map = {self.vocab[i]: i for i in range(len(self.vocab))}
        self.col_idx_map.update({EOS: len(self.vocab), UNIGRAM: len(self.vocab) + 1, BT_BEGIN: len(self.vocab) + 2})

    def _fill_bt_begin(self):
        index = 0
        arr = []
        while index < len(self.matrix):
            arr.append(len([e for e in self.matrix[index] if e != 0]))
            index += 1
        return arr

    def _fill_bt_end(self):
        arr = []
        index = 0
        while index < len(self.matrix):
            count = 0
            for i in range(len(self.matrix)):
                if self.matrix[i][index] != 0:
                    count += 1
            arr.append(count)
            index += 1
        return arr

    def count_bigrams(self):
        # the additional thing is to only set the additional information Kneyser-Ney is using.
        self.matrix = np.array(
            [[0.0 for _ in range(len(self.vocab) + 1)] for _ in range(len(self.vocab) + 1)])
        for sent in self.training_data:
            for i in range(len(sent) - 1):
                bigram = (sent[i], sent[i + 1])
                self.matrix[self.row_idx_map[bigram[0]]][self.col_idx_map[bigram[1]]] += 1

        self.unigrams = self.unigram_model.counts
        self.unigrams.update({BOS: len(self.training_data)})
        self.bt_begin = self._fill_bt_begin()
        self.bt_end = self._fill_bt_end()

    """
    Fills every cell in the table with it's recalculated probability using the formula 
    """

    def _fill_cell(self, unigrams, bt_begin, bt_end, T, w2, w1, row_idx, col_idx):
        idx_row, idx_col = row_idx[w1], col_idx[w2]

        # count of bigram
        C_w1_w2 = T[idx_row][idx_col]
        # discount coeff
        d = 0.75
        # count of w1
        C_w1 = unigrams[w1]

        # discount coeff
        lambd = (0.75 / C_w1) * bt_begin[idx_row]

        # probability of continuation word
        cont = bt_end[idx_col] / sum(bt_end)

        # formula together
        kn_formula = (max(C_w1_w2 - d, 0) / C_w1) + lambd * cont
        T[idx_row][idx_col] = kn_formula

    def kneyser_ney_probs(self, unigrams, bt_begin, bt_end, row_idx, col_idx):
        for i in tqdm(range(len(self.matrix))):
            if i >= len(self.vocab):
                for j in range(len(self.matrix)):
                    if j >= len(self.vocab):
                        self._fill_cell(unigrams, bt_begin, bt_end, self.matrix, EOS, BOS, row_idx, col_idx)
                    else:
                        self._fill_cell(unigrams, bt_begin, bt_end, self.matrix, self.vocab[j], BOS, row_idx, col_idx)
            else:
                for j in range(len(self.matrix)):
                    if j >= len(self.vocab):
                        self._fill_cell(unigrams, bt_begin, bt_end, self.matrix, EOS, self.vocab[i], row_idx, col_idx)
                    else:
                        self._fill_cell(unigrams, bt_begin, bt_end, self.matrix, self.vocab[j], self.vocab[i], row_idx,
                                        col_idx)
            time.sleep(0.1)

        self.probs = np.log10(self.matrix)

    def train(self, train_file):
        unigram = Unigram(frequency_threshold=self.frequency_threshold)
        unigram.train(train_file=train_file)

        self.unigram_model = unigram
        self.vocab = unigram.vocab

        self.load_and_preprocess(train=True)
        self.set_mappings()
        self.count_bigrams()
        self.kneyser_ney_probs(self.unigrams, self.bt_begin, self.bt_end, self.row_idx_map, self.col_idx_map)

    def perplexity(self, test_sentences):
        logprobs, N = 0, 0
        for sent in test_sentences:
            N += len(sent) - 1
            logprobs += sum([self.probs[self.row_idx_map[sent[i]]][self.col_idx_map[sent[i + 1]]] for i in
                             range(len(sent) - 1)])
        return 10 ** (-logprobs / N)

    def test(self, test_file):
        self.unigram_model.load_and_preprocess(corpus_file=test_file, train=False)
        self.load_and_preprocess(train=False)
        self.pp = self.perplexity(self.test_data)
        return self.pp

    def compress(self):

        return {"Perplexity": self.pp, "Row2IdMap": self.row_idx_map, "Col2IdMap": self.col_idx_map,
                "Table": self.probs.tolist()}


if __name__ == '__main__':
    bigram = Bigram(frequency_threshold=3)
    bigram.train(train_file="../data/SherlockHolmes-train.txt")
    bigram.test(test_file="../data/SherlockHolmes-test.txt")
    bigram.compress()
