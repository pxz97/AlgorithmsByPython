# 文本相似度/匹配模型归纳总结
在自然语言处理（Natural Language Processing，NLP）领域中，经常会涉及到如何度量两个文本间相似度的问题。
例如在对话系统（Dialog system）和信息检索（Information retrieval）等任务中，文本相似度的匹配尤为重要。
比如，基于聚类算法发现微博热点话题时，我们需要度量每个文本之间的相似度，然后将内容足够相似的微博内容聚成
一个簇；在问答系统中，我们会准备一些经典问题的对应答案，当用户的问题与经典问题很相似时，系统会直接返回准备
好的答案；在对语料库进行预处理时，我们需要基于文本的相似度，把重复的文本给挑出来并删掉。

总之，文本相似度算法一直是自然语言处理领域中非常重要的一个分支，
在此部分中，我将会对文本相似度/匹配方法进行总结，由于本人实力有限，如有错误，恳请大家指正！

综合现有的文本相似度匹配方法，可以大致将其分为以下三类：

1. 基于关键词
2. 基于向量空间
3. 基于深度学习

### 1 基于关键词分类
#### 1.1 n-gram相似度
n-gram是一种基于统计语言模型的算法。它的思想是将文本里面的内容按照字词进行大小为n的滑动窗口操作，
每一个字词片段称为一个gram。这种切分方式非常简单：使用一个长度为n的窗口，从左到右、逐字符滑动；每一步都会框到一个
字符串，即一个gram。如表中所示，是几种常见的n-gram切分效果实例。

|取值|名称|切分实例|
|-----|-----|-----|
|1|unigram|文/本/相/似/度/匹/配|
|2|bigram|文本/本相/相似/似度/度匹/匹配|
|3|trigram|文本相/本相似/相似度/似度匹/度匹配|

n-gram相似度计算方法是指按长度n将原句划分为词段，然后对两个句子A和B，根据共有的词段数量去定义两个句子的相似度：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/ngram.png)

其中，GN(A)和GN(B)分别表示句子A和句子B中n-gram的集合。Similarity的值越小，表示两者越相似，当Similarity为0时，两者完全相等。
#### 1.2 Jaccard相似度
Jaccard相似度就是计算两个句子之间字词集合的并集与交集的比值。值越大，表示两个句子越相似，公式如下：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/jaccard.png)
### 2 基于向量空间
#### 2.1 TF-IDF
TF-IDF可以分为两部分：词频（Term Frequency，TF）和逆文本频率（Inverse Document Frequency，IDF）。TF-IDF是一种统计方法，用以评估一个字词对于一个文档集或一个语料库中其中一份文档的重要程度。它的核心思想是：字词的重要性随着它在文档中出现的次数成正比而增加，随着它在语料库中出现的频率成反比而下降。

词频（TF）表示关键字词在文本中出现的频率。公式如下：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/tf.png)

其中，TF_ij表示词i在文档j中的TF值，n_ij表示该词在文档j中出现的次数，sum_nkj表示文档j中所有词出现的次数之和。

逆向文档频率（IDF）表示关键字词在语料库中出现的普遍程度。公式如下：

![imag](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/idf.png)

其中，|D|表示语料库中文档总数，|j:ti∈dj|表示包含词语ti的文档数目。

实际的TF-IDF值为词频（TF）和逆向文档频率（IDF）的乘积，即：

![imag](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/tf-idf.png)

通过计算文档中每个字词的TF-IDF值，可以得到一个表征文本的向量，通过计算其相似度值（余弦相似度），就可以计算两个文本的相似度。
#### 2.2 Word2Vec
早在2003年Bengio等人提出的文章[A neural probabilistic language model](./Paper/NLP/【NNLM】 A neural probabilistic language model.pdf)中提出的神经网络模型（NNLM）就用到了Embedding词向量，只是在这个模型中，目标是生成语言模型，词向量只是一个副产品。

Mikolov等人与2013年在文章[Efficient Estimation of Word Representations in Vector Space](./Paper/NLP/【Word2Vec】 Efficient Estimation of Word Representations in Vector Space (Google 2013).pdf)中提出的Word2Vec的核心思想是认为词的意思可以从它频繁出现的上下文信息中体现。

Word2Vec又可以分为两种结构，一种是CBOW，即利用窗口内的上下文去预测中心词；另一种是Skip-gram，利用中心词去预测窗口内的上下文。CBOW和Skip-gram结构如下图所示：

![imag](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/word2vec.png)

无论是CBOW还是Skip-gram，都是有“输入层+隐含层+输出层”组成，其中输入层和输出层的维度都为词表大小V，隐含层维度为词向量维度K，则输入层与隐含层之间的权重为V×K的矩阵，隐含层与输出层之间的权重为K×V的矩阵。模型输入的是字词的one-hot向量表示。

本文中没有展示出CBOW和Skip-gram的具体训练细节和tricks。总之最终训练好模型后，取出模型的权重矩阵，就可以将稀疏的one-hot向量映射为稠密的Embedding向量。在得到两个文本的词向量表示之后，我们就可以通过计算相似度的方法（例如余弦相似度），计算两个文本之间的相似度。

### 3 基于深度学习
随着深度学习技术的发展，其在图像和语音识别等领域取得了很大的成就，近些年来，深度学习在自然语言处理领域的应用也取得了不小的成就。

深度学习技术的一个优点是不依赖于人为地设计特征，通过其自身的深度结构可以从文本中提取深层的语义特征。语义相似工作已经逐渐从人工设计特征转向了分布式表达和神经网络结构相结合的方式。（Recent work has moved away from hand crafted features and towards modeling with distributed representations and neural network architectures.）

在本部分中，我将分别介绍三种经典流行的深度学习模型，分别是DSSM、ESIM和目标在NLP领域大行其道的Bert模型。

#### 3.1 DSSM
DSSM有Po-Sen Huang等人于2013年在文章[Learning deep structured semantic models for web search using clickthrough data](./Paper/NLP/【DSSM】 Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.pdf)中提出。最初应用于检索场景下的query和doc匹配。在DSSM之前，更多是利用一些传统的机器学习算法。所以DSSM可以说是深度学习在文本相似度匹配领域的一个先驱者。

DSSM模型中利用点击率来代替相关性，点击数据中包含了大量用户的问句和对应的点击文档。

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/dssm.png)

上图为采用DNN结构的DSSM。DSSM可以分为三个部分：Embedding层，特征提取层和输出层，我将针对这三个部分分别进行描述。
#### 3.1.1 Embedding层
Embedding层对应图中的Term Vector和Word Hashing。Term Vector将文本转化为向量值，论文中作者采用了Bag-of-words，即词袋模型。

之所以会用到Word Hashing层，是因为主流文本转向量方法更多是采用Embedding，而这种方法一个很大的缺陷就是会出现OOV问题（Out-of-vocabulary，即当用到另一个数据集时，这个数据集中出现了一些现有的词表中不存在的词）。对此问题，作者提出了word hashing方法，一方面可以降低输入数据的维度，另一方面也可以保证不会出现OOV问题。
具体以good这个单词为例，分为三步：
- 在good两端添加临界符变为#good#；

- 采用前面提到的n-gram将其分为多部分，如果是trigrams，则结果是[#go,goo,ood,od#]；

- 最终将good转化为[#go,goo,odd,od#]的向量表示。

但是这种方法只在英文中有效，并不适合中文，所以可以采用字向量的格式，将中文的最小粒度看作字，或者增大语料库，最大可能地降低OOV问题出现的可能。

#### 3.1.2 特征提取层
特征提取层的结构比较简单，即三层激活函数为tanh的全连接层，通过三层全连接层提取深层特征，最终得到query和doc两个128维的semantic feature，并进行了余弦相似度的计算。

#### 3.1.3 输出层
输出层的结构也比较简单，即一个softmax层，将计算相似度看作是一个二分类问题，即点击则为1，未点击则为0，则输出的点击概率就可以用来表征query和doc之间的相似度。

#### 3.1.4 优缺点
DSSM模型有以下几个优点：
- 解决了传统方法中存在的OOV问题；
- 使用有监督方法，优化了语义embedding中的映射问题；
- 省去了人工提取特征的过程。

但是DSSM也有很明显的缺点：
- word hashing可能造成冲突；
- DSSM采用了词袋模型，损失了上下文信息。

针对DSSM中存在的问题，渐渐有人提出了很多优化变种，其中较为经典的便是[CNN-DSSM](./Paper/NLP/【CNN-DSSM】 A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval.pdf)、[LSTM-DSSM](./Paper/NLP/[LSTM-DSSM] Semantic Modelling With Long-short-term Memory for Information Retrieval.pdf
)、[MV-DSSM](./Paper/NLP/【MV-DSSM】 A Multi-Viev Deep Learning Approach for Cross Domain User Model in Recommendation Systems.pdf)等。

### 3.2 ESIM
前面介绍了可以称作文本匹配先驱者的DSSM模型之后，接下来将要介绍的是被称作当年最强的短文本匹配模型——ESIM。

ESIM提出于文章[Enhanced LSTM for Natural Language Inference](./Paper/NLP/【ESIM】 Enhanced LSTM for Natural Language Inference.pdf)中。ESIM主要用作文本推理，给定一个前提Premise推导出假设Hypothesis，其损失函数的目标是判断Premise和Hypothesis是否有关联，即是否可以由Premise推导出Hypothesis，因此可以用作文本匹配模型。如下图左半部所示，ESIM主要分为四部分：Input Encoding、Local Inference Modeling、Interence Composition和Prediction。

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim1.png)

#### 3.2.1 Input Encoding
Input Encoding阶段对输入通过Embedding+BiLSTM进行编码。BiLSTM可以学习到一句话中word和它上下文的关系，可以理解为word embedding得到稠密词向量之后，在当前语境情况下进行重新编码，得到新的Embedding向量。

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim2.png)

#### 3.2.2 Local Inference Modeling
在Local Inference阶段，首先对两句话做alignment，计算两个句子word之间的相似度，得到相似度矩阵：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim7.png)

在得到两句话word之间的相似度矩阵之后，分别对两句话进行local inference modeling，用前面得到的相似度矩阵，结合a、b两句话，互相生成彼此相似性加权之后的特征，且保持特征尺寸不变：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim3.png)

在local inference之后，进行enhancement of local inference information，即计算a和align之后的a的差和点积。

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim4.png)

#### 3.2.3 Inference Composition
在此阶段，将前一步得到的a和b再次利用BiLSTM提取上下文信息，这里的BiLSTM和前面的作用不太一样，此处的BiLSTM主要用于捕获局部推理信息：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim5.png)

然后分别用MaxPooling和AvgPooling进行池化，将最终得到的特征拼接：

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/esim6.png)

#### 3.2.4 Prediction
这部分将前面得到的特征置入全连接层，全连接层的激活函数为tanh，最终通过softmax归一化得到最终的结果。

ESIM具有以下优点：
- 精细地设计序列式的推断结构；
- 考虑到了局部推断和全局推断。

ESIM通过句子间的注意力机制（intra-sentence attention），来实现局部的推断，并进一步实现全局的推断。

### 3.3 Bert
在Google与2017年提出[Transformer](./Papar/NLP/【Transformer】 Attention Is All You Need (google 2017).pdf)，并于2018年提出[Bert]([BERT] Bert Pre-training of Deep Bidirectional Transformers for Language Understanding (google 2019).pdf)，Bert在NLP几乎刷新了所有领域的精度，几乎起到了统治性的地位。Bert主要具有以下几个特点：
- 使用了Transformer中的Encoder编码器作为主体框架，Transformer可以很有效地捕捉语句中的双向关系；
- 使用了Mask Language Model（MLM）和Next Sentence（NSP）作为训练模型的任务；
- 使用了大量的语料作为模型的训练数据。

#### 3.3.1 网络结构
Bert的结构其实比较好理解，就是Transformer中的Encoder结构叠加而成。

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/transformer.png)

上图为Transformer的结构，而Bert所用到的是图中左半部分中灰色方框中的Encoder结构。正如Transformer原文标题“Attention is all you need”所表达的意思，Transformer的核心就是“attention”。

#### 3.3.2 输入表示
Bert的输入向量是3个嵌入特征的对应加和，如下图所示，这三个特征分别是：
- WordPiece embedding：WordPiece将单词划分为一组有限的公共子词单元，如下图中所示，“playing”被划分为“play”和“##ing”；
- Position embedding：RNN和LSTM这样的模型是串行地处理数据的，其独特的计算方式使得其得到的特征天然的带有序列顺序信息，而Bert是并行处理数据的，得到的特征中并不会包含序列顺序信息，因此引入了position embedding，人为地添加位置信息；
- Segment embedding：用来标识区分两个句子，例如我们所讲的相似度计算任务，模型的输入是两个句子，则第一个句子的segment embedding为0，第二个句子的segment embedding为1。

![image](https://github.com/pxz97/AlgorithmsByPython/blob/master/figure/bert_input.png)

#### 3.3.3 预训练任务
Bert的训练涉及到两个任务：Masked Language Model（MLM）和Next Sentence Prediction（NSP）。

MLM任务指在训练时随机从文本中mask掉一些词，通过上下文去预测被mask掉的词。在Bert的原文中，WordPiece Token中有15%被随机mask掉，而这被mask掉的15%的WordPiece Token中，80%被替换为特殊符号[mask]，10%被替换为其它任何单词，10%保留原始的Token。

NSP任务指判断输入数据中的B句是否为A句的下一句。是下一句则输出“IsNext”，反之输出“NotText”。这种训练方式和相似度匹配的训练方式非常像，都是从语料库中挑选相关联的两句作为正类，不相关的两句作为负类。

#### 3.3.4 微调
为了实现Bert相应的下游任务，并不是拿到相应的语料库之后，将Bert模型从头开始训练。事实上，Bert的预训练过程非常受计算资源的限制，论文中提到的Bert Large在64块TPU芯片上训练了4天。而Google开源了其所训练的各种型号的模型参数，我们只需要在Google提供的Bert模型基础上，设计相关下游任务的输出结构，便可以完成对特定任务的微调。

对于相似度匹配任务，可以将其看作二分类任务，在Bert的后面加上softmax即可实现相似度计算的任务。

### 3.4 Simbert
#TODO
