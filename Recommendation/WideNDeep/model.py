import tensorflow as tf
from tensorflow.keras import layers
from Recommendation.base.base_model import BaseModel


class WideNDeep(BaseModel):
    def build(self):
        categorical_columns = []
        for feature, vocab in self.GENRE_FEATURES.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab
            )
            emb_col = tf.feature_column.embedding_column(cat_col, 10)
            categorical_columns.append(emb_col)

        movie_col = tf.feature_column.categorical_column_with_identity(key="movieId", num_buckets=1001)
        movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
        categorical_columns.append(movie_emb_col)

        user_col = tf.feature_column.categorical_column_with_identity(key="userId", num_buckets=30001)
        user_emb_col = tf.feature_column.embedding_column(user_col, 10)
        categorical_columns.append(user_emb_col)

        numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                             tf.feature_column.numeric_column('movieRatingCount'),
                             tf.feature_column.numeric_column('movieAvgRating'),
                             tf.feature_column.numeric_column('movieRatingStddev'),
                             tf.feature_column.numeric_column('userRatingCount'),
                             tf.feature_column.numeric_column('userAvgRating'),
                             tf.feature_column.numeric_column('userRatingStddev')]

        rated_movie = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
        crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([movie_col, rated_movie], 10000))

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

        deep = layers.DenseFeatures(numerical_columns + categorical_columns)(inputs)
        deep = layers.Dense(128, activation="relu")(deep)
        deep = layers.Dense(128, activation="relu")(deep)

        wide = layers.DenseFeatures(crossed_feature)(inputs)
        both = layers.concatenate([deep, wide])
        outputs = layers.Dense(1, activation="sigmoid")(both)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC"), tf.keras.metrics.AUC(curve="PR")]
        )

        return model
