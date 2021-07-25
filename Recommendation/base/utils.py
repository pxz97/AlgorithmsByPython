import tensorflow as tf


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=128,
        label_name="label",
        na_value="0",
        num_epochs=1,
        ignore_errors=True
    )
    return dataset
