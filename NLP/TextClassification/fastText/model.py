from tensorflow.keras import layers
from tensorflow.keras import Sequential
from NLP.TextClassification.base.base_model import BaseModel


class FastText(BaseModel):
    def __init__(self, params, generator, vocab_generator):
        super().__init__()
        self.params = params
        self.generator = generator
        self.vocab_generator = vocab_generator
        self.model = self.build()

    def build(self):
        model = Sequential([
            layers.Embedding(self.params["vocab_size"], self.params["embed_size"]),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.params["num_classes"], activation="softmax")
        ])

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model
