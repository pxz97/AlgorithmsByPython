from tensorflow.keras import layers


class BaseModel:

    def __init__(self, params):
        self._params = params

    def make_embedding_layer(self, name="embedding", embed_type="char", **kwargs):

        def init_embedding(weights=None):

            if embed_type == "char":
                input_dim = self._params["vocab_size_char"]
                output_dim = self._params["embed_size_char"]
            elif embed_type == "word":
                input_dim = self._params["vocab_size_word"]
                output_dim = self._params["embed_size_word"]
            else:
                raise ValueError("Your input embedding type is wrong, it must be 'word' or 'char'")

            return layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                trainable=False,
                name=name,
                weights=weights
            )

        if embed_type == "char":
            embed_weights = self._params["embedding_matrix_char"]
        elif embed_type == "word":
            embed_weights = self._params["embedding_matrix_word"]
        else:
            raise ValueError("Your input embedding type is wrong, it must be 'word' or 'char'")

        if not embed_weights:
            embedding = init_embedding()
        else:
            embedding = init_embedding(weights=[embed_weights])

        return embedding

    def _make_multi_layer_perceptron_layer(self) -> layers.Layer:

        def _warpper(x):

            activation = self._params["mlp_activation"]

            for _ in range(self._params["mlp_layers_num"]):
                x = layers.Dense(units=self._params["mlp_units_num"],
                                 activation=activation)(x)

            return layers.Dense(units=self._params["mlp_out_num"],
                                activation=activation)

        return _warpper

    def _make_inputs(self) -> list:

        input_left = layers.Input(
            name="text_left",
            shape=self._params["input_shapes"][0]
        )

        input_right = layers.Input(
            name="text_right",
            shape=self._params["input_shapes"][1]
        )

        return [input_left, input_right]

    def _make_output_layer(self) -> layers.Layer:

        task = self._params["task"]

        if task == "classification":
            return layers.Dense(units=2, activation="softmax")
        elif task == "ranking":
            return layers.Dense(units=1, activation="linear")
        else:
            raise ValueError(f"{task} is not a valid task type."
                             f"Must be in 'ranking' or 'classification'.")

    def _create_base_network(self):

        def _wrapper(x):

            pass

        return _wrapper

    def build(self):

        pass
