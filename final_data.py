import pandas as pd
import tensorflow as tf
import numpy as np
import h5py
CSV_COLUMN_NAMES = list(map(lambda x:'feature'+str(x),range(1,6813)))+['label']
COLUMN_NAMES = list(map(lambda x:'feature'+str(x),range(1,6813)))

def load_data(y_name='label', train_fraction=0.7, seed=None):
    f = h5py.File('train_data.mat','r')
    print('loading raw data...')
    feats = f.get('train_feat')
    labels = f.get('train_label')
    print('processing feats...')
    train = np.array(feats).T
    print('processing labels...')
    labels = np.array(labels).T

    train = pd.DataFrame(train,columns=COLUMN_NAMES)
    train.insert(6812,'label',labels)

    # split into trainset and test set
    print('splitting...')
    x_train = train.sample(frac=train_fraction, random_state=seed)
    x_test = train.drop(x_train.index)

    y_train = x_train.pop(y_name).astype(int)
    y_test = x_test.pop(y_name).astype(int)
    print('test and train data loaded.')
    return (x_train, y_train), (x_test, y_test)

def load_pred():
    
    return 0

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


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
# 

# load_data()