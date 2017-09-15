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
        dataset = dataset.shuffled()
        dataset = dataset.translated_labels()

        train, test = dataset.split(0.8)
        train.to_tfrecord(train_path)
        test.to_tfrecord(test_path)

    return train_path, test_path

def extract_configs(path):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    return model_config, train_config, input_config

def train_model(train, test, model_config, train_config, input_config):
    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    create_input_dict_fn = functools.partial(input_reader_builder.build, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1

    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                job_name=task_info.type,
                                task_index=task_info.index)

        if task_info.type == 'ps':
            server.join()
            exit()

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    num_clones = 1
    clone_on_cpu = False
    train_dir = 'models'

    trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                num_clones, worker_replicas, clone_on_cpu, ps_tasks,
                worker_job_name, is_chief, train_dir)


if __name__ == '__main__':
    train_tfrecord, test_tfrecord = ensure_tfrecord(train_tfrecord, test_tfrecord)
    model_config, train_config, input_config = extract_configs(config_file)
    train_model(train_tfrecord, test_tfrecord, model_config, train_config, input_config)


