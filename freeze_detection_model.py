import tensorflow as tf

from google.protobuf import text_format
from object_detection.exporter import export_inference_graph
from object_detection.protos import pipeline_pb2

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile('data/detection/faces.config', 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
export_inference_graph('image_tensor', pipeline_config, 'data/models/detection/model.ckpt-94328', 'data/pbs', True, 'face_detection')