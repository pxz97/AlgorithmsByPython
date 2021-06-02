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

        """
        将输入QKV分别传入线性层中
        做完线性变换后，开始为每个头分割输入，使用view方法对线性变换结果进行维度重塑
        每个头可以获得一部分词特征组成的句子
        为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到语义与句子位置的关系
        """
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        """
        多头注意力机制后，得到了每个头计算结果组成的4维张量，将其转换为输入的形状
        先对第二、三维转置（对第一步处理环节的逆操作）
        """
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终多头注意力结构的输出
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.s2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features: 词嵌入的维度
        :param eps: 一个足够小的数，在规范化公式的分母中出现，防止分母为0
        """
        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2和b2，a2为全1张量，b2为全0张量
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        """
        首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致
        接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果
        最后对结果乘以缩放参数，即a2，加上位移参数b2
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        :param size: 词嵌入维度大小
        :param dropout: 对模型中节点随机抑制比率
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度大小
        :param self_attn: 多头自注意力子层
        :param feed_forward: 前馈全连接层
        :param dropout: 置0比率
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入的维度大小
        :param self_attn: 多头自注意力对象，Q=K=V
        :param src_attn: 多头注意力对象，Q!=K=V
        :param feed_forward: 前馈全连接层对象
        :param dropout: 置0比率
        """
        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层输入
        :param memory: 来自编码器层的语义存储变量memory
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        """
        m = memory

        """
        将x传入第一个子层结构，第一个子层结构的输入分别是x和self_attn函数，因为是自注意力机制，所以Q，K，V相等
        target_mask是目标数据掩码张量，因为此时模型可能还没有生成任何目标数据，所以要对目标数据进行遮掩
        比如
            在编码器准备生成第一个字符时，其实已经传入了第一个字符以便计算损失
            但是我们不希望生成第一个字符时利用这个信息，因此会将其遮掩
            同样生成第二个字符时，模型只能使用第一个字符信息，不能使用第二个字符及之后的信息
        """
        x = self.sublayer[0](x, lambda y: self.self_attn(x, x, x, target_mask))

        """
        第二层为常规的注意力机制，q是输入x，k和v是编码层输出memory
        source_mask并非为了抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，以此来提升模型效果和训练速度
        """
        x = self.sublayer[1](x, lambda y: self.src_attn(x, m, m, source_mask))

        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 输出部分的类别生成器对象
        """
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        # 使用src_embed对source做处理，然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        # 使用tgt_embed对target做处理，然后和source_mask，target_mask，memory一起传给
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    :param source_vocab: 源数据特征（词汇）总数
    :param target_vocab: 目标数据特征（词汇）总数
    :param N: 编码器和解码器堆叠数
    :param d_model: 词向量映射维度
    :param d_ff: 前馈全连接网络中变换矩阵的维度
    :param head: 多头注意力结构中的头数
    :param dropout: 置零比率
    """
    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    """
    最外层是EncoderDecoder，在EncoderDecoder中
    分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构
    目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层
    在编码器层中有attention子层以及前馈全连接子层
    在解码器层中有两个attention子层以及前馈全连接层
    """
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    # 参数初始化：一旦参数的维度大于1，则将其初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


sourcd_vocab = 11
target_vocab = 11
N = 6

if __name__ == "__main__":
    res = make_model(sourcd_vocab, target_vocab, N)
    print(res)

