import tensorflow as tf
from Recommendation.base.utils import get_dataset
from Recommendation.base.conf import train_samples_file_path, test_samples_file_path

if __name__ == "__main__":
    model_name = "wide_deep"

    train_dataset = get_dataset(train_samples_file_path)
    test_dataset = get_dataset(test_samples_file_path)

    if model_name == "deepfm":
        from Recommendation.DeepFM.model import DeepFM
        model = DeepFM().build()
    elif model_name == "neuralcf":
        from Recommendation.NeuralCF.model import NeuralCF
        model = NeuralCF(hidden_units=[10, 10], arch=1).build()
    elif model_name == "embedding_mlp":
        from Recommendation.EmbeddingMLP.model import EmbeddingMLP
        model = EmbeddingMLP().build()
    elif model_name == "wide_deep":
        from Recommendation.WideNDeep.model import WideNDeep
        model = WideNDeep().build()
    else:
        raise ValueError

    model.summary()

    model.fit(train_dataset, epochs=5)

    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print("\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}.".format(test_loss, test_accuracy,
                                                                                        test_roc_auc, test_pr_auc))

    predictions = model.predict(test_dataset)
    for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ",
              ("Good Rating" if bool(goodRating) else "Bad Rating"))
