import os

root_path = os.path.dirname(os.path.dirname(__file__))


base_params = {
    "vocab_size": 10000,
    "embed_size": 200,
    "embedding_matrix": [],
    "max_len": 64,
    "num_classes": 15,
    "dropout_rate": 0.2,
    "bert_dict": root_path + "/chinese_L-12_H-768_A-12/vocab.txt",
    "bert_config": root_path + "/chinese_L-12_H-768_A-12/bert_config.json",
    "bert_checkpoint": root_path + "/chinese_L-12_H-768_A-12/bert_model.ckpt",
}