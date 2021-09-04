from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras import layers
from tensorflow.keras import Model


class BERT():

    def __init__(self, params, generator, vocab_generator):
        self.params = params
        self.dict_path = self.params["bert_dict"]
        self.config_path = self.params["bert_config"]
        self.checkpoint_path = self.params["bert_checkpoint"]
        self.generator = generator
        self.vocab_generator = vocab_generator
        self.model = self.build()

    def build(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        input1 = layers.Input(shape=(None, ))
        input2 = layers.Input(shape=(None, ))
        fea = bert_model([input1, input2])
        fea = layers.Lambda(lambda x: x[:, 0])(fea)
        outputs = layers.Dense(self.params["num_classes"], activation="softmax")(fea)
        model = Model([input1, input2], outputs)

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model


