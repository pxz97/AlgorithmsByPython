import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K


class KMaxPooling(Layer):

    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__()

        self.input_spec = InputSpec(ndim=2)
        self.k = k

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k

    def call(self, inputs, **kwargs):
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=True, name=None)[0]

        return top_k

    def get_config(self):

        config = {"k": self.k}
        base_config = super(KMaxPooling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class MatchingLayer(Layer):

    def __init__(self, normalize: bool = False, matching_type: str = "dot", **kwargs):
        super(MatchingLayer, self).__init__()

        self._normalize = normalize
        self._validate_matching_type(matching_type)
        self._matching_type = matching_type
        self._shape1 = None
        self._shape2 = None

    @classmethod
    def _validate_matching_type(cls, matching_type: str = "dot"):

        valid_matching_type = ["dot", "mul", "plus", "minus", "concat"]

        if matching_type not in valid_matching_type:
            raise ValueError(f"{matching_type} is not a valid matching type, "
                             f"{valid_matching_type} expected.")

    def build(self, input_shape: list):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("A 'MatchingLayer' layer should be called on a list of 2 inputs")

        self._shape1 = input_shape[0]
        self._shape2 = input_shape[1]

        for idx in 0, 2:
            if self._shape1[idx] != self._shape2[idx]:
                raise ValueError(f"Incompatible dimensions: "
                                 f"{self._shape1[idx]} != {self._shape2[idx]}."
                                 f"Layer shapes: {self._shape1}, {self._shape2}.")

    def call(self, inputs: list, **kwargs):

        x1 = inputs[0]
        x2 = inputs[1]

        if self._matching_type == "dot":
            if self._normalize:
                x1 = tf.math.l2_normalize(x1, axis=2)
                x2 = tf.math.l2_normalize(x2, axis=2)
            return tf.expand_dims(tf.einsum("abd,acd->abc", x1, x2), 3)
        else:
            x1_exp = tf.stack([x1] * self._shape2[1], 2)
            x2_exp = tf.stack([x2] * self._shape1[1], 1)

            if self._matching_type == "mul":
                def func(x, y):
                    return x * y
            elif self._matching_type == "plus":
                def func(x, y):
                    return x + y
            elif self._matching_type == "minus":
                def func(x, y):
                    return x - y
            elif self._matching_type == "concat":
                def func(x, y):
                    return tf.concat([x, y], axis=3)
            else:
                raise ValueError(f"Invalid matching type."
                                 f"{self._matching_type} received."
                                 f"Must be in 'dot', 'mul', 'plus', 'minus' and 'concat'.")

            return func(x1_exp, x2_exp)

    def compute_output_shape(self, input_shape: list) -> tuple:

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("A 'MatchingLayer' layer should be called on a list of 2 inputs.")

        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError("A 'MatchingLayer' layer should be called on 2 inputs with 3 dimensions.")

        if shape1[0] != shape2[0] or shape1[2] != shape2[2]:
            raise ValueError("A 'MatchingLayer' layer should be called on 2 inputs with same 0,2 dimensions.")

        if self._matching_type in ["mul", "plus", "minus"]:
            return shape1[0], shape1[1], shape2[1], shape1[2]
        elif self._matching_type == "dot":
            return shape1[0], shape1[1], shape2[1], 1
        elif self._matching_type == "concat":
            return shape1[0], shape1[1], shape2[1], shape1[2] + shape2[2]
        else:
            raise ValueError(f"Invalid 'matching_type'."
                             f"{self._matching_type} received."
                             f"Must be in 'mul', 'plus', 'minus', 'dot', and 'concat'.")

    def get_config(self) ->dict:

        config = {
            "normalize": self._normalize,
            "matching_type": self._matching_type,
        }
        base_config = super(MatchingLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class MultiPerspective(Layer):

    def __init__(self, mp_dim, epsilon=1e-6, **kwargs):
        super(MultiPerspective, self).__init__()

        self.mp_dim = mp_dim
        self.epsilon = epsilon
        self.strategy = 4

    def build(self, input_shape):
        super(MultiPerspective, self).build(input_shape)

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        embedding_size = int(input_shape[-1] / 2)

        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.mp_dim, embedding_size * 2 * self.strategy),
                                      initializer="glorot_uniform",
                                      trainable=True)
        self.kernel_full_fw = self.kernel[:, :embedding_size]
        self.kernel_full_bw = self.kernel[:, embedding_size: embedding_size * 2]
        self.kernel_attentive_fw = self.kernel[:, embedding_size * 2: embedding_size * 3]
        self.kernel_attentive_bw = self.kernel[:, embedding_size * 3: embedding_size * 4]
        self.kernel_max_attentive_fw = self.kernel[:, embedding_size * 4: embedding_size * 5]
        self.kernel_max_attentive_bw = self.kernel[:, embedding_size * 5: embedding_size * 6]
        self.kernel_max_pool_fw = self.kernel[:, embedding_size * 6: embedding_size * 7]
        self.kernel_max_pool_bw = self.kernel[:, embedding_size * 7]
        self.built = True

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        return input_shape[0], input_shape[1], self.mp_dim * 2 * self.strategy

    def get_config(self):

        config = {
            "mp_dim": self.mp_dim,
            "epsilon": self.epsilon
        }
        base_config = super(MultiPerspective, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):

        h1 = inputs[0]
        h2 = inputs[1]

        embedding_size = int(K.int_shape(h1)[-1] / 2)

        h1_fw = h1[:, :, :embedding_size]
        h1_bw = h1[:, :, embedding_size]
        h2_fw = h2[:, :, :embedding_size]
        h2_bw = h2[:, :, embedding_size]

        list_matching = []

        matching_fw = self._full_matching(h1_fw, h2_fw, self.kernel_full_fw)
        matching_bw = self._full_matching(h1_bw, h2_bw, self.kernel_full_bw)
        list_matching.extend([matching_fw, matching_bw])

        cosine_matrix_fw = self._cosine_matrix(h1_fw, h2_fw)
        cosine_matrix_bw = self._cosine_matrix(h1_bw, h2_bw)

        matching_fw = self._attentive_matching(h1_fw, h2_fw, cosine_matrix_fw, self.kernel_attentive_fw)
        matching_bw = self._attentive_matching(h1_bw, h2_bw, cosine_matrix_bw, self.kernel_attentive_bw)
        list_matching.extend([matching_fw, matching_bw])

        matching_fw = self._max_pooling_matching(h1_fw, h2_fw, self.kernel_max_pool_fw)
        matching_bw = self._max_pooling_matching(h1_bw, h2_bw, self.kernel_max_pool_bw)
        list_matching.extend([matching_fw, matching_bw])

        return K.concatenate(list_matching, axis=-1)

    def _cosine_similarity(self, x1, x2):

        cos = K.sum(x1 * x2, axis=-1)
        x1_norm = K.sqrt(K.maximum(K.sum(K.square(x1), axis=-1), self.epsilon))
        x2_norm = K.sqrt(K.maximum(K.sum(K.square(x2), axis=-1), self.epsilon))

        return cos / x1_norm / x2_norm

    def _cosine_matrix(self, x1, x2):

        x1 = K.expand_dims(x1, axis=2)
        x2 = K.expand_dims(x2, axis=1)
        cos_matrix = self._cosine_similarity(x1, x2)

        return cos_matrix

    def _mean_attentive_vectors(self, x2, cosine_matrix):

        expanded_cosine_matrix = K.expand_dims(cosine_matrix, axis=-1)
        x2 = K.expand_dims(x2, axis=1)
        weighted_sum = K.sum(expanded_cosine_matrix * x2, axis=2)
        sum_cosine = K.expand_dims(K.sum(cosine_matrix, axis=-1) + self.epsilon, axis=-1)
        attentive_vector = weighted_sum / sum_cosine

        return attentive_vector

    def _max_attentive_vector(self, x2, cosine_matrix):

        max_x2_step = K.argmax(cosine_matrix, axis=-1)

        embedding_size = K.int_shape(x2)[-1]
        time_steps = K.int_shape(max_x2_step)[-1]
        if time_steps is None:
            time_steps = K.shape(max_x2_step)[-1]

        x2 = K.reshape(x2, (-1, embedding_size))
        max_x2_step = K.reshape(max_x2_step, (-1, ))
        max_x2 = K.gather(x2, max_x2_step)
        attentive_vector = K.reshape(max_x2, K.stack([-1, time_steps, embedding_size]))

        return attentive_vector

    def _time_distributed_multiply(self, x, w):

        n_dim = K.ndim(x)
        embedding_size = K.int_shape(x)[-1]
        time_steps = K.int_shape(x)[1]
        if time_steps is None:
            time_steps = K.shape(x)[1]

        x = K.reshape(x, (-1, embedding_size))
        x = K.expand_dims(x, axis=1)
        w = K.expand_dims(w, axis=0)
        x = x * w

        if n_dim == 3:
            x = K.reshape(x, K.stack([-1, time_steps, self.mp_dim, embedding_size]))
            x.set_shape([None, None, None, embedding_size])
        elif n_dim == 2:
            x = K.reshape(x, K.stack([-1, self.mp_dim, embedding_size]))
            x.set_shape([None, None, embedding_size])

        return x

    def _full_matching(self, h1, h2, w):

        h2_last_state = h2[:, -1, :]
        h1 = self._time_distributed_multiply(h1, w)
        h2 = self._time_distributed_multiply(h2_last_state, w)
        h2 = K.expand_dims(h2, axis=1)
        matching = self._cosine_similarity(h1, h2)

        return matching

    def _max_pooling_matching(self, h1, h2, w):

        h1 = self._time_distributed_multiply(h1, w)
        h2 = self._time_distributed_multiply(h2, w)
        h1 = K.expand_dims(h1, axis=2)
        h2 = K.expand_dims(h2, axis=1)
        cos = self._cosine_similarity(h1, h2)
        matching = K.max(cos, axis=2)

        return matching

    def _attentive_matching(self, h1, h2, cosine_matrix, w):

        h1 = self._time_distributed_multiply(h1, w)
        attentive_vec = self._mean_attentive_vectors(h2, cosine_matrix)
        attentive_vec = self._time_distributed_multiply(attentive_vec, w)
        matching = self._cosine_similarity(h1, attentive_vec)

        return matching

    def _max_attentive_matching(self, h1, h2, cosine_matrix, w):

        h1 = self._time_distributed_multiply(h1, w)
        max_attentive_vec = self._max_attentive_vector(h2, cosine_matrix)
        max_attentive_vec = self._time_distributed_multiply(max_attentive_vec, w)
        matching = self._cosine_similarity(h1, max_attentive_vec)

        return matching


class SoftAttention:

    def __call__(self, inputs):

        inputs_a = inputs[0]
        inputs_b = inputs[1]

        attention = layers.Lambda(self.__attention,
                                  output_shape=self._attention_output_shape,
                                  arguments=None)(inputs)

        align_a = layers.Lambda(self._soft_alignment,
                                output_shape=self._soft_alignment_output_shape,
                                arguments=None)([attention, inputs_a])
        align_b = layers.Lambda(self._soft_alignment,
                                output_shape=self._soft_alignment_output_shape,
                                arguments=None)([attention, inputs_b])

        return align_a, align_b

    def _attention(self, inputs):

        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1],
                                                          pattern=(0, 2, 1)))

        return K.permute_dimensions(attn_weights, (0, 2, 1))

    def _attention_output_shape(self, inputs):

        input_shape = inputs[0]
        embedding_size = input_shape[1]

        return input_shape[0], embedding_size, embedding_size

    def _soft_alignment(self, inputs):

        attention = inputs[0]
        sentence = inputs[1]

        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    def _soft_alignment_output_shape(self, inputs):

        attention_shape = inputs[0]
        sentence_shape = inputs[1]

        return attention_shape[0], attention_shape[1], sentence_shape[2]


class DynamicPoolingLayer(Layer):

    def __init__(self, psize1, psize2, **kwargs):
        super(DynamicPoolingLayer, self).__init__()

        self._psize1 = psize1
        self._psize2 = psize2

    def build(self, input_shape):
        super(DynamicPoolingLayer, self).build()

        input_shape_one = input_shape[0]
        self._msize1 = input_shape_one[1]
        self._msize2 = input_shape_one[2]

    def call(self, inputs, **kwargs):

        self._validate_dpool_size()
        x, dpool_index = inputs
        dpool_shape = tf.shape(dpool_index)
        batch_index_one = tf.expand_dims(
            tf.expand_dims(
                tf.range(dpool_shape[0]), axis=-1),
            axis=-1)
        batch_index = tf.expand_dims(
            tf.tile(batch_index_one, [1, self._msize1, self._msize2]),
            axis=-1)
        dpool_index_ex = tf.concat([batch_index, dpool_index], axis=3)
        x_expand = tf.gather_nd(x, dpool_index_ex)
        stride1 = self._msize1 // self._psize1
        stride2 = self._msize2 // self._psize2

        x_pool = tf.nn.max_pool(x_expand,
                                [1, stride1, stride2, 1],
                                [1, stride1, stride2, 1],
                                "VALID")

        return x_pool

    def compute_output_shape(self, input_shape):

        input_shape_one = input_shape[0]

        return None, self._psize1, self._psize2, input_shape_one[3]

    def get_config(self) -> dict:

        config = {
            "psize1": self._psize1,
            "psize2": self._psize2
        }
        base_config = super(DynamicPoolingLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def _validate_dpool_size(self):

        suggestion = self.get_size_suggestion(
            self._msize1, self._msize2, self._psize1, self._psize2
        )
        if suggestion != (self._psize1, self._psize2):
            raise ValueError(
                "DynamicPooling Layer can not "
                f"generate ({self._psize1} x {self._psize2}) output "
                f"feature map, please use ({suggestion[0]} x {suggestion[1]})"
                f" instead. 'model.params['dpool_size'] = {suggestion}' "
            )

    @classmethod
    def get_size_suggestion(cls, msize1, msize2, psize1, psize2):

        stride1 = msize1 // psize1
        stride2 = msize2 // psize2
        suggestion1 = msize1 // stride1
        suggestion2 = msize2 // stride2

        return suggestion1, suggestion2
