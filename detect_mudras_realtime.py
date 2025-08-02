import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# Load model
PATH_TO_MODEL_DIR = 'content/exported_model/saved_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

# Load label map
category_index = label_map_util.create_category_index_from_labelmap('label_map.pbtxt')

def detect_mudras_realtime():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detection_classes = detections['detection_classes'].astype(np.int64)

        viz_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            detections['detection_boxes'],
            detection_classes,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=0.8
        )

        cv2.imshow('Mudra Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_mudras_realtime()
