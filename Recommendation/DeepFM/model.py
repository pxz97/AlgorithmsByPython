import tensorflow as tf
from tensorflow.keras import layers
from Recommendation.base.base_model import BaseModel


class DeepFM(BaseModel):
    def __init__(self):
        super(DeepFM, self).__init__()

    def build(self):
        inputs = {
            "movieId": layers.Input(name="movieId", shape=(), dtype="int32"),
            "userId": layers.Input(name="userId", shape=(), dtype="int32"),

            "movieAvgRating": layers.Input(name="movieAvgRating", shape=(), dtype="float32"),
            "movieRatingStddev": layers.Input(name="movieRatingStddev", shape=(), dtype="float32"),
            "movieRatingCount": layers.Input(name="movieRatingCount", shape=(), dtype="int32"),
            "userAvgRating": layers.Input(name="userAvgRating", shape=(), dtype="float32"),
            "userRatingStddev": layers.Input(name="userRatingStddev", shape=(), dtype="float32"),
            "userRatingCount": layers.Input(name="userRatingCount", shape=(), dtype="int32"),
            "releaseYear": layers.Input(name="releaseYear", shape=(), dtype="int32"),

            "userGenre1": layers.Input(name="userGenre1", shape=(), dtype="string"),
            "userGenre2": layers.Input(name="userGenre2", shape=(), dtype="string"),
            "userGenre3": layers.Input(name="userGenre3", shape=(), dtype="string"),
            "userGenre4": layers.Input(name="userGenre4", shape=(), dtype="string"),
            "userGenre5": layers.Input(name="userGenre5", shape=(), dtype="string"),
            "movieGenre1": layers.Input(name="movieGenre1", shape=(), dtype="string"),
            "movieGenre2": layers.Input(name="movieGenre2", shape=(), dtype="string"),
            "movieGenre3": layers.Input(name="movieGenre3", shape=(), dtype="string"),

            "userRatedMovie1": layers.Input(name="userRatedMovie1", shape=(), dtype="int32"),
        }

        user_col = tf.feature_column.categorical_column_with_identity(key="userId", num_buckets=30001)
        user_emb_col = tf.feature_column.embedding_column(user_col, 10)
        user_ind_col = tf.feature_column.indicator_column(user_col)

        movie_col = tf.feature_column.categorical_column_with_identity(key="movieId", num_buckets=1001)
        movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
        movie_ind_col = tf.feature_column.indicator_column(movie_col)

        user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                                   vocabulary_list=self.genre_vocab)
        user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, 10)
        user_genre_ind_col = tf.feature_column.indicator_column(user_genre_col)

        item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                                   vocabulary_list=self.genre_vocab)
        item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, 10)
        item_genre_ind_col = tf.feature_column.indicator_column(item_genre_col)

        fm_first_order_columns = [movie_ind_col, user_ind_col, user_genre_ind_col, item_genre_ind_col]

        deep_feaature_columns = [tf.feature_column.numeric_column("releaseYear"),
                                 tf.feature_column.numeric_column("movieRatingCount"),
                                 tf.feature_column.numeric_column("movieAvgRating"),
                                 tf.feature_column.numeric_column("movieRatingStddev"),
                                 tf.feature_column.numeric_column("userRatingCount"),
                                 tf.feature_column.numeric_column("userAvgRating"),
                                 tf.feature_column.numeric_column("userRatingStddev"),
                                 user_emb_col,
                                 movie_emb_col]

        user_emb_layer = layers.DenseFeatures([user_emb_col])(inputs)
        item_emb_layer = layers.DenseFeatures([movie_emb_col])(inputs)
        user_genre_emb_layer = layers.DenseFeatures([user_genre_emb_col])(inputs)
        item_genre_emb_layer = layers.DenseFeatures([item_genre_emb_col])(inputs)

        fm_first_order_layer = layers.DenseFeatures(fm_first_order_columns)(inputs)

        product_layer_item_user = layers.Dot(axes=1)([item_emb_layer, user_emb_layer])
        product_layer_item_genre_user_genre = layers.Dot(axes=1)([item_genre_emb_layer, user_genre_emb_layer])
        product_layer_item_genre_user = layers.Dot(axes=1)([item_genre_emb_layer, user_emb_layer])
        product_layer_user_genre_item = layers.Dot(axes=1)([item_emb_layer, user_genre_emb_layer])

        deep = layers.DenseFeatures(deep_feaature_columns)(inputs)
        deep = layers.Dense(64, activation="relu")(deep)
        deep = layers.Dense(64, activation="relu")(deep)

        concat_layer = layers.concatenate(
            [fm_first_order_layer, product_layer_item_user, product_layer_item_genre_user_genre,
             product_layer_item_genre_user, product_layer_user_genre_item, deep], axis=1)

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(concat_layer)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
        )

        return model





