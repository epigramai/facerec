import functools
import json
import os
import tensorflow as tf

from google.protobuf import text_format
from object_detection import trainer
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

from tfwrapper.datasets import FDDB

train_tfrecord = os.path.join('data', 'detection', 'train.tfrecord')
test_tfrecord = os.path.join('data', 'detection', 'test.tfrecord')
config_file = os.path.join('data', 'detection', 'faces.config')

def ensure_tfrecord(train_path, test_path):
    if not os.path.isfile(train_path):
        dataset = FDDB(size=1000)
        dataset = dataset.translated_labels()

        train, test = dataset.split(0.8)
        train.to_tfrecord(train_path)
        test.to_tfrecord(test_path)

    return train_path, test_path


if __name__ == '__main__':
    train_tfrecord, test_tfrecord = ensure_tfrecord(train_tfrecord, test_tfrecord)


