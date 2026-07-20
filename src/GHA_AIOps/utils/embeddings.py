import torch

SPECIAL_SYMBOLS = ["<PAD>", "<UNK>", "<SOS>", "\n"]


class CharVocab:
    def __init__(self):
        self.str2int = {}

        for i, c in enumerate(SPECIAL_SYMBOLS):
            self.str2int[c] = i

        for i in range(32, 127):
            self.str2int[chr(i)] = i - 32 + len(SPECIAL_SYMBOLS)
        self.int2str = dict((v, k) for k, v in self.str2int.items())

    def encode(self, text):
        encoded = []
        for c in text:
            if c in self.str2int:
                encoded.append(self.str2int[c])
            else:
                encoded.append(self.str2int["<UNK>"])
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indexes):
        decoded = ''
        for index in indexes:
            index = int(index)
            if index in self.int2str:
                decoded += self.int2str[index]
            else:
                decoded += self.int2str[0]
        return decoded

    def __len__(self):
        return len(self.str2int)
