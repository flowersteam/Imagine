import numpy as np
from abc import ABC, abstractmethod


def analyze_descr(descriptions):
    '''
    Create vocabulary + extract all descriptions split and in lower case
    '''
    split_descriptions = []
    word_list = []
    max_sequence_length = 0
    for descr in descriptions:
        split_descr = descr.lower().split(' ')
        len_descr = len(split_descr)
        if len_descr > max_sequence_length:
            max_sequence_length = len_descr
        word_list.extend(split_descr)
        split_descriptions.append(split_descr)

    word_set = set(word_list)

    return split_descriptions, max_sequence_length, word_set


class Vocab(object):
    '''
    Vocabulary class:
    id2word: mapping of index to word
    word2id mapping of words to index
    '''

    def __init__(self, words):
        word_list = sorted(list(set(words)))
        self.id2word = dict(zip([0] + [i + 1 for i in range(len(word_list))], ['pad'] + word_list))
        self.size = len(word_list) + 1  # +1 to account for padding
        self.word2id = dict(zip(['pad'] + word_list, [0] + [i + 1 for i in range(len(word_list))]))


class AbstractEncoder(ABC):
    '''
    Encoder must implement function encode and decode and be init with vocab and max_seq_length
    '''

    def __init__(self, vocab, max_seq_length):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        super().__init__()

    @abstractmethod
    def encode(self, description):
        pass

    @abstractmethod
    def decode(self, sequence):
        pass


class OneHotEncoder(AbstractEncoder):

    def _word2one_hot(self, word):
        id = self.vocab.word2id[word]
        out = np.zeros(self.vocab.size)
        out[id] = 1
        return out

    def encode(self, split_description):
        one_hot_seq = []
        for word in split_description:
            one_hot_seq.append(self._word2one_hot(word))
        while len(one_hot_seq) < self.max_seq_length:
            one_hot_seq.append(np.zeros(self.vocab.size))
        return one_hot_seq

    def decode(self, one_hot_seq):
        words = []
        for vect in one_hot_seq:
            if np.sum(vect) > 0:
                words.append(self.vocab.id2word[np.where(vect > 0)[0][0]])
        return ' '.join(words)


class IdEncoder(AbstractEncoder):

    def encode(self, split_description):
        id_seq = []
        for word in split_description:
            id_seq.append(self.vocab.word2id[word])
        while len(id_seq) < self.max_seq_length:
            id_seq.append(0)
        return id_seq

    def decode(self, id_seq):
        words = []
        for id in id_seq:
            if id > 0:
                words.append(self.vocab.id2word[id])
        return ' '.join(words)
