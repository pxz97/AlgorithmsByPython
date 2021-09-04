from tensorflow.keras import Model
from tensorflow.keras import layers


class TextCNN:

    def __init__(self, params, generator, vocab_generator):
        self.params = params
        self.generator = generator
        self.vocab_generator = vocab_generator
        self.model = self.build()

    def build(self):
        inputs = layers.Input(shape=(self.params["max_len"], ), dtype="float64")
        embd = layers.Embedding(input_dim=self.params["vocab_size"],
                                output_dim=self.params["embed_size"],
                                trainable=False,
                                weights=self.params["embedding_matrix"])(inputs)
        conv1 = layers.Conv1D(256, 3, padding="valid", strides=1, activation="relu")(embd)
        conv1 = layers.MaxPool1D(pool_size=48)(conv1)
        conv2 = layers.Conv1D(256, 4, padding="valid", strides=1, activation="relu")(embd)
        conv2 = layers.MaxPool1D(pool_size=47)(conv2)
        conv3 = layers.Conv1D(256, 5, padding="valid", strides=1, activation="relu")(embd)
        conv3 = layers.MaxPool1D(pool_size=46)(conv3)
        concat = layers.concatenate([conv1, conv2, conv3], axis=-1)
        flat = layers.Flatten()(concat)
        drop = layers.Dropout(self.params["dropout_rate"])(flat)
        outputs = layers.Dense(self.params["num_classes"])(drop)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model



