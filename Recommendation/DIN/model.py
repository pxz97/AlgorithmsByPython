import tensorflow as tf
from tensorflow.keras import layers
from Recommendation.base.base_model import BaseModel


class DIN(BaseModel):

    def __init__(self):
        super(DIN, self).__init__()

        self.RECENT_MOVIES = 5
        self.EMBEDDING_SIZE = 10

    def build(self):
        inputs = {
            'movieAvgRating': layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
            'movieRatingStddev': layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
            'movieRatingCount': layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
            'userAvgRating': layers.Input(name='userAvgRating', shape=(), dtype='float32'),
            'userRatingStddev': layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
            'userRatingCount': layers.Input(name='userRatingCount', shape=(), dtype='int32'),
            'releaseYear': layers.Input(name='releaseYear', shape=(), dtype='int32'),

            'movieId': layers.Input(name='movieId', shape=(), dtype='int32'),
            'userId': layers.Input(name='userId', shape=(), dtype='int32'),
            'userRatedMovie1': layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),
            'userRatedMovie2': layers.Input(name='userRatedMovie2', shape=(), dtype='int32'),
            'userRatedMovie3': layers.Input(name='userRatedMovie3', shape=(), dtype='int32'),
            'userRatedMovie4': layers.Input(name='userRatedMovie4', shape=(), dtype='int32'),
            'userRatedMovie5': layers.Input(name='userRatedMovie5', shape=(), dtype='int32'),

            'userGenre1': layers.Input(name='userGenre1', shape=(), dtype='string'),
            'userGenre2': layers.Input(name='userGenre2', shape=(), dtype='string'),
            'userGenre3': layers.Input(name='userGenre3', shape=(), dtype='string'),
            'userGenre4': layers.Input(name='userGenre4', shape=(), dtype='string'),
            'userGenre5': layers.Input(name='userGenre5', shape=(), dtype='string'),
            'movieGenre1': layers.Input(name='movieGenre1', shape=(), dtype='string'),
            'movieGenre2': layers.Input(name='movieGenre2', shape=(), dtype='string'),
            'movieGenre3': layers.Input(name='movieGenre3', shape=(), dtype='string'),
        }

        user_col = tf.feature_column.categorical_column_with_identity(key="userId", num_buckets=30001)
        user_emb_col = tf.feature_column.embedding_column(user_col, self.EMBEDDING_SIZE)

        user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                                   vocabulary_list=self.genre_vocab)
        user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, self.EMBEDDING_SIZE)

        item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                                   vocabulary_list=self.genre_vocab)
        item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, self.EMBEDDING_SIZE)

        candidate_movie_col = [tf.feature_column.numeric_column(key="movieId", default_value=0)]

        recent_rate_col = [
            tf.feature_column.numeric_column(key="userRatedMovie1", default_value=0),
            tf.feature_column.numeric_column(key="userRatedMovie2", default_value=0),
            tf.feature_column.numeric_column(key="userRatedMovie3", default_value=0),
            tf.feature_column.numeric_column(key="userRatedMovie4", default_value=0),
            tf.feature_column.numeric_column(key="userRatedMovie5", default_value=0),
        ]

        user_profile = [
            user_emb_col,
            user_genre_emb_col,
            tf.feature_column.numeric_column("userRatingCount"),
            tf.feature_column.numeric_column("userAvgRating"),
            tf.feature_column.numeric_column("userRatingStddev"),
        ]

        context_features = [
            item_genre_emb_col,
            tf.feature_column.numeric_column("releaseYear"),
            tf.feature_column.numeric_column("movieRatingCount"),
            tf.feature_column.numeric_column("movieAvgRating"),
            tf.feature_column.numeric_column("movieRatingStddev"),
        ]

        candidate_layer = layers.DenseFeatures(candidate_movie_col)(inputs)
        user_behavior_layer = layers.DenseFeatures(recent_rate_col)(inputs)
        user_profile_layer = layers.DenseFeatures(user_profile)(inputs)
        context_features_layer = layers.DenseFeatures(context_features)(inputs)

        movie_emb_layer = layers.Embedding(input_dim=1001, output_dim=self.EMBEDDING_SIZE, mask_zero=True)

        user_behavior_emb_layer = movie_emb_layer(user_behavior_layer)

        candidate_emb_layer = movie_emb_layer(candidate_layer)
        candidate_emb_layer = tf.squeeze(candidate_emb_layer, axis=1)

        repeated_candidate_emb_layer = layers.RepeatVector(self.RECENT_MOVIES)(candidate_emb_layer)

        activation_sub_layer = layers.Subtract()([user_behavior_emb_layer,
                                                  repeated_candidate_emb_layer])

        activation_product_layer = layers.Multiply()([user_behavior_emb_layer,
                                                      repeated_candidate_emb_layer])

        activation_all = layers.concatenate([activation_sub_layer, user_behavior_emb_layer,
                                             repeated_candidate_emb_layer, activation_product_layer], axis=-1)

        activation_unit = layers.Dense(32)(activation_all)
        activation_unit = layers.PReLU()(activation_unit)
        activation_unit = layers.Dense(1, activation="sigmoid")(activation_unit)
        activation_unit = layers.Flatten()(activation_unit)
        activation_unit = layers.RepeatVector(self.EMBEDDING_SIZE)(activation_unit)
        activation_unit = layers.Permute((2, 1))(activation_unit)
        activation_unit = layers.Multiply()([user_behavior_emb_layer, activation_unit])

        user_behaviors_pooled_layers = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(activation_unit)

        concat_layer = layers.concatenate([user_profile_layer, user_behaviors_pooled_layers,
                                           candidate_emb_layer, context_features_layer])
        outputs = layers.Dense(128)(concat_layer)
        outputs = layers.PReLU()(outputs)
        outputs = layers.Dense(64)(outputs)
        outputs = layers.PReLU()(outputs)
        outputs = layers.Dense(1, activation="sigmoid")(outputs)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC"), tf.keras.metrics.AUC(curve="PR")]
        )

        return model
