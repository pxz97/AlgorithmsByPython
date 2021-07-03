import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_positional_encoding(seq_len, embed_dim):
    """
    位置编码
    seq_len: 最大序列长度
    embed_dim: 字嵌入维度
    """
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(seq_len)
    ])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])
    return positional_encoding


positional_encoding = get_positional_encoding(seq_len=100, embed_dim=10)
plt.figure(figsize=(10, 10))
# sns.heatmap(positional_encoding)
plt.title("Sinusoidal Function")
plt.xlabel("hidden dimension")
plt.ylabel("sequence length")
plt.show()
