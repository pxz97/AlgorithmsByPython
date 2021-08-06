from tensorflow.keras import layers, Model
from NLP.TextMatching.base.base_model import BaseModel
from NLP.TextMatching.base.base_layer import SoftAttention


class ESIM(BaseModel):

    def __init__(self):
        super(ESIM, self).__init__()

        self._params["mlp_layers_num"] = 1
        self._params["mlp_units_num"] = 256
        self._params["mlp_fan_out_num"] = 128
        self._params["lstm_units"] = 64
        self._params["dropout_rate"] = 0.3
        self._params["embed_size"] = 100

    def build(self):

        input_a, input_b = self._make_inputs()

        # ---------- Embedding Layers ----------
        embedding = self.make_embedding_layer()

        embedded_a = embedding(input_a)
        embedded_b = embedding(input_b)

        # ---------- Encoding Layers ----------
        bilstm = layers.Bidirectional(layers.LSTM(
            units=self._params["lstm_units"],
            return_sequences=True,
            dropout=self._params["dropoout_rate"]
        ))

        encoded_a = bilstm(embedded_a)
        encoded_b = bilstm(embedded_b)

        # ---------- Local Inference Layers ----------
        atten_a, atten_b = SoftAttention()([encoded_a, encoded_b])

        sub_atten_a = layers.Lambda(lambda x: x[0] - x[1])([encoded_a, atten_a])
        sub_atten_b = layers.Lambda(lambda x: x[0] - x[1])([encoded_b, atten_b])

        mul_atten_a = layers.Lambda(lambda x: x[0] * x[1])([encoded_a, atten_a])
        mul_atten_b = layers.Lambda(lambda x: x[0] * x[1])([encoded_b, atten_b])

        m_a = layers.concatenate([encoded_a, atten_a, sub_atten_a, mul_atten_a])
        m_b = layers.concatenate([encoded_b, atten_b, sub_atten_b, mul_atten_b])

        # ---------- Inference Composition Layer ----------
        composition_a = layers.Bidirectional(layers.LSTM(
            units=self._params["lstm_units"],
            return_sequences=True,
            dropout=self._params["dropout_rate"]
        ))(m_a)

        avg_pool_a = layers.GlobalAveragePooling1D()(composition_a)
        max_pool_a = layers.GlobalMaxPooling1D()(composition_a)

        composition_b = layers.Bidirectional(layers.LSTM(
            units=self._params["lstm_units"],
            return_sequences=True,
            dropout=self._params["dropout_rate"]
        ))(m_b)

        avg_pool_b = layers.GlobalAveragePooling1D()(composition_b)
        max_pool_b = layers.GlobalMaxPooling1D()(composition_b)

        pooled = layers.concatenate([avg_pool_a, max_pool_a, avg_pool_b, max_pool_b])
        pooled = layers.Dropout(rate=self._params["dropout_rate"])(pooled)

        # ---------- Classification Layer ----------
        mlp = self._make_multi_layer_perceptron_layer()(pooled)
        mlp = layers.Dropout(rate=self._params["dropout_rate"])(mlp)

        outputs = self._make_output_layer()(mlp)

        model = Model(inputs=[input_a, input_b], outputs=outputs)

        return model
