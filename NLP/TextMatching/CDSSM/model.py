from NLP.TextMatching.base.base_model import BaseModel
from tensorflow.keras import layers
from tensorflow.keras import Model


class CDSSM(BaseModel):

    def _create_base_network(self):

        def _wrapper(x):

            x = self.embedding()(x)
            x = layers.Conv1D(filters=self._params["filters"],
                              kernel_size=self._params["kernel_size"],
                              strides=self._params["strides"],
                              padding=self._params["padding"],
                              activation=self._params["conv_activation"],
                              kernel_initializer=self._params["w_initializer"],
                              bias_initializer=self._params["b_initializer"])(x)
            x = layers.Dropout(self._params["dropout_rate"])(x)
            x = layers.GlobalMaxPooling1D()(x)
            x = self._make_multi_layer_perceptron_layer()(x)
            return x
        return _wrapper

    def build(self):

        self.embedding = self.make_embedding_layer()
        base_network = self._create_base_network()
        input_left, input_right = self._make_inputs()
        x = [base_network(input_left),
             base_network(input_right)]
        x = layers.Dot(axes=[1, 1], normalize=True)(x)
        outputs = self._make_output_layer()(x)
        model = Model(inputs=[input_left, input_right],
                      outputs=outputs)
        return model
