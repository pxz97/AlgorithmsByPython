import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embeddings(nn.Module):
    """
    定义Embeddings类来实现文本嵌入层，这里s说明代表两各一模一样的嵌入层，它们共享参数
    """
    def __init__(self, d_model, vocab):
        """
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: 输入给该模型的文本通过词汇映射后的张量
        """
        return self.lut(x) * math.sqrt(self.d_model)  # math.sqrt(self.d_model)起到一个缩放的作用


class PositionalEncoding(nn.Module):
    """
    因为在Transformer的编码器结构中，并没有针对词汇位置信息的处理，因此需要在Embedding层后加入位置编码器，将词汇位置不同
    可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: 词嵌入维度
        :param dropout: 置0比率
        :param max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，一个0矩阵，矩阵大小为max_len * d_model
        pe = torch.zeros(max_len, d_model)

        """
        初始化一个绝对位置矩阵，在这里，词汇的绝对位置就是用它的索引来表示
        首先使用arange方法获得一个连续自然数向量，然后使用unsqueeze方法扩展向量维度
        因为传入的参数为1，代表矩阵扩展的位置，会使向量变成一个max_len * 1的矩阵
        """
        position = torch.arange(0, max_len).unsqueeze(1)

        """
        绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中
        最简单的思路就是先将max_len * 1的绝对位置矩阵，变换成max_len * d_model形状，然后覆盖原来的初始位置编码矩阵即可
        要做这种变换，就需要一个1 * d_model形状的变换矩阵div_term，对这个变换矩阵的要求除了形状外，
        还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快地收敛
        """
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        """
        把pe位置矩阵注册成模型的buffer
        虽然pe对模型效果是有帮助的，但是不需要进行优化更新
        注册之后就可以在模型保存后重加载时和模型结构与参数一同被加载
        """
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        需要对pe做一些适配操作，将这个三维张量的第二位（即句子最大长度）将切片到与输入的x的第二维相同即x.size(1)
        因为max_len为5000，一般来讲实在太大了，很难有一个句子包含5000个词汇，所以要进行与输入张量的适配
        """
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    """
    :param size: 掩码张量最后两个维度的大小，最后两维形成一个方阵
    """
    attn_shape = (1, size, size)

    # np.triu: 形成上三角矩阵
    subsequent_masks = np.triu(np.ones(attn_shape), k=1).astype("uint8")

    """
    将numpy类型转化为torch中的tensor，内部做一个1 - 的操作
    其实是做了一个三角阵的反转，subsequent_masks中的每个元素都会被1减
    如果是0，subsequent_masks中该位置由0变1
    如果是1，subsequent_masks中该位置由1变0
    """
    return torch.from_numpy(1 - subsequent_masks)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # 取query最后一维的大小，一般情况下就等同于词嵌入维度

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 使用tensor的masked_fill，将掩码张量和scores张量每个位置一一比较，如果掩码张量处为0
        # 则对应的scores张量用-1e9来替换
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """
    用于生成相同网络层的克隆函数
    :param module: 需要克隆的目标网络层
    :param N: 需要克隆的数量
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head  # 得到每个头获得的分割词向量维度d_k
        self.head = head  # 传入头数h
        # 内部变换矩阵为embedding_dim * embedding_dim，四个矩阵分别为：Q，K，V和最后的拼接矩阵
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None  # 最后得到的注意力张量
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)






