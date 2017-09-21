import os
import tensorflow as tf

DEFAULT_DETECTION_PB_PATH = 'data/pbs/face_detection_graph.pb'

def _find_latest_checkpoint_id(path):
    folder = '/'.join(path.split('/')[:-1])
    filename = path.split('/')[-1]

    ids = []
    for f in os.listdir(folder):
        if f.startswith(filename) and f.endswith('.index'):
            ids.append(f.split('-')[1].split('.')[0])

    return sorted(ids)[-1]

class FaceDetector():
    def __init__(self, path=DEFAULT_DETECTION_PB_PATH):
        with tf.Session() as sess:
            with tf.gfile.FastGFile(path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def,name='')
                
                self.graph = sess.graph
                self.image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
                self.boxes_tensor = sess.graph.get_tensor_by_name('detection_boxes:0')
                self.scores_tensor = sess.graph.get_tensor_by_name('detection_scores:0')
                self.classes_tensor = sess.graph.get_tensor_by_name('detection_classes:0')


    def detect(self, img):
        with tf.Session(graph=self.graph) as sess:
            tensors = [self.boxes_tensor, self.scores_tensor, self.classes_tensor]
            boxes, scores, classes = sess.run(tensors, feed_dict={self.image_tensor: [img]})
            print('Boxes: ' + str(boxes))
            print('Scores: ' + str(scores))
            print('Classes: ' + str(classes))

        return boxes[0][:10]