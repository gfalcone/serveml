from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import mlflow
import argparse
import pandas as pd
import tensorflow as tf
import mlflow.tensorflow


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(args):
    with mlflow.start_run(experiment_id=4):
        # Fetch the data
        (train_x, train_y), (test_x, test_y) = load_data()

        # Feature columns describe how to use the input.
        my_feature_columns = []
        for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Two hidden layers of 10 nodes each.
        hidden_units = [10, 10]

        # Build 2 hidden layer DNN with 10, 10 units respectively.
        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            hidden_units=hidden_units,
            # The model must choose between 3 classes.
            n_classes=3)

        # Train the Model.
        classifier.train(
            input_fn=lambda:train_input_fn(train_x, train_y,
                                                     args.batch_size),
            steps=args.train_steps)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:eval_input_fn(test_x, test_y,
                                                    args.batch_size))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        # Creating output tf.Variables to specify the output of the saved model.
        feat_specifications = {
            'SepalLength': tf.Variable([], dtype=tf.float64, name="SepalLength"),
            'SepalWidth':  tf.Variable([], dtype=tf.float64, name="SepalWidth"),
            'PetalLength': tf.Variable([], dtype=tf.float64, name="PetalLength"),
            'PetalWidth': tf.Variable([], dtype=tf.float64, name="PetalWidth")
        }

        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_specifications)
        temp = tempfile.mkdtemp()

        classifier.export_saved_model(
            temp,
            receiver_fn,
        ).decode("utf-8")

        # custom code for registering models
        mlflow_client = mlflow.tracking.MlflowClient('http://localhost:5000')
        run_id = mlflow_client.list_run_infos(experiment_id=4)[0].run_id

        mlflow.register_model(
            "runs:/{}/artifacts/model".format(run_id),
            "tensorflow_model"
        )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
