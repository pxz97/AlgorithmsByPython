import numpy as np
from tensorflow.keras.utils import to_categorical
import codecs
from keras_bert import Tokenizer


def seq_padding(x, padding):
    return np.array([np.concatenate([i, [0] * (padding - len(i))]) if len(i) < padding else i[:padding] for i in x])


def onehot(y, num_classes):
    y = to_categorical(y, num_classes=num_classes)
    return y


class BERT_Generator:
    def __init__(self, data, dict_path, label_dict, max_len, batch_size=1, is_bert=False):
        self.data = data
        self.batch_size = batch_size
        self.dict_path = dict_path
        self.label_dict = label_dict
        self.is_bert = is_bert
        self.max_len = max_len
        self.tokenizer = self.get_token_dict()

    def get_token_dict(self):
        self.token_dict = {}
        with codecs.open(self.dict_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        tokenizer = Tokenizer(self.token_dict)
        return tokenizer

    def __iter__(self):
        if self.is_bert:
            while True:
                idxs = np.array(len(self.data))
                np.random.shuffle(idxs)
                x_l1, x_l2, y_l = [], [], []
                for i in idxs:
                    text = self.data[i][0]
                    x1, x2 = self.tokenizer.encode(first=text)
                    x_l1.append(x1)
                    x_l2.append(x2)
                    y_l.append(self.data[i][1])
                    if self.batch_size == len(x_l1) or idxs[-1] == i:
                        X1 = seq_padding(x_l1, self.max_len)
                        X2 = seq_padding(x_l2, self.max_len)
                        y_index = [[self.label_dict.get(i)] for i in y_l]
                        y_one_hot = onehot(y_index, len(self.label_dict))
                        yield [X1, X2], y_one_hot
                        x_l1, x_l2, y_l = [], [], []
