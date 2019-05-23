import os
import torch

import codecs
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        with codecs.open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with codecs.open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def safe_tokenize(self, path):
        with codecs.open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
        # Tokenize file content
        with codecs.open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    else:
                        ids[token] = self.dictionary.word2idx['<unk>']
                        token += 1

        return ids

    def safe_tokenize_lines(self, path):
        all_lines = []
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                tokens = len(words)

                ids = torch.LongTensor(tokens)
                token = 0

                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    else:
                        ids[token] = self.dictionary.word2idx['<unk>']
                        token += 1

                all_lines.append(ids)

        return all_lines

    def safe_tokenize_sentence(self, line):
        words = line.split() + ['<eos>']
        tokens = len(words)

        ids = torch.LongTensor(tokens)
        token = 0

        words = line.split() + ['<eos>']
        for word in words:
            if word in self.dictionary.word2idx:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
            else:
                ids[token] = self.dictionary.word2idx['<unk>']
                token += 1

        return ids
