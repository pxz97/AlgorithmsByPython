import tensorflow as tf
from tensorflow.keras import layers
from Recommendation.base.base_model import BaseModel


class NeuralCF(BaseModel):

    def __init__(self, hidden_units, arch=1):
        super(NeuralCF, self).__init__()
        self.arch = arch
        self.hidden_units = hidden_units

    def build(self):
        inputs = {
            "movieId": layers.Input(name="movieId", shape=(), dtype="int32"),
            "userId": layers.Input(name="userId", shape=(), dtype="int32"),
        }

        movie_col = tf.feature_column.categorical_column_with_identity(key="movieId", num_buckets=1001)
        movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

        user_col = tf.feature_column.categorical_column_with_identity(key="userId", num_buckets=30001)
        user_emb_col = tf.feature_column.embedding_column(user_col, 10)

        item_tower = layers.DenseFeatures(movie_emb_col)(inputs)
        user_tower = layers.DenseFeatures(user_emb_col)(inputs)

        if self.arch == 1:
            interact_layer = layers.concatenate([item_tower, user_tower])
            for num_nodes in self.hidden_units:
                interact_layer = layers.Dense(num_nodes, activation="relu")(interact_layer)

            outputs = layers.Dense(1, activation="sigmoid")(interact_layer)

            model = tf.keras.Model(inputs, outputs)
        elif self.arch == 2:
            for num_nodes in self.hidden_units:
                item_tower = layers.Dense(num_nodes, activation="relu")(item_tower)
                user_tower = layers.Dense(num_nodes, activation="relu")(user_tower)

            outputs = layers.Dot(axes=1)([item_tower, user_tower])
            outputs = layers.Dense(1, activation="sigmoid")(outputs)

            model = tf.keras.Model(inputs, outputs)
        else:
            raise ValueError("The value of arch must be 1 or 2.")

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC"), tf.keras.metrics.AUC(curve="PR")]
        )

        return model
