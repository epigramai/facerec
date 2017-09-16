import os
import tensorflow as tf

DEFAULT_DETECTION_CHECKPOINT_PATH = 'data/models/detection/model.ckpt'

def _find_latest_checkpoint_id(path):
    folder = '/'.join(path.split('/')[:-1])
    filename = path.split('/')[-1]

    ids = []
    for f in os.listdir(folder):
        if f.startswith(filename) and f.endswith('.index'):
            ids.append(f.split('-')[1].split('.')[0])

    return sorted(ids)[-1]

class FaceDetector():
    def __init__(self, checkpoint_path=DEFAULT_DETECTION_CHECKPOINT_PATH):
        latest_checkpoint_id = _find_latest_checkpoint_id(checkpoint_path)
        checkpoint_path = '-'.join([checkpoint_path, latest_checkpoint_id])
        metagraph_path = '.'.join([checkpoint_path, 'meta'])
        
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(metagraph_path)
            saver.restore(sess, checkpoint_path)
            self.graph = sess.graph

    def detect(self, img):
        return img