import os
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # train.txt文件id化后的张量
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            # 64-bit integer (signed) |	torch.LongTensor |  torch.cuda.LongTensor
            # torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。
            ids = torch.LongTensor(tokens)  # tokens维张量
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    # note：不是字典，是张量
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
