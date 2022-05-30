from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from reef_net.preprocess import resize_and_pad_image
import tensorflow as tf
from reef_net.utils import swap_xy

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def visualize_detections(
    image, boxes, classes, scores, fname, figsize=(24, 16), linewidth=1, color=[0, 0, 1],
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize, frameon=False)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()

    for box, _cls, score in zip(boxes, classes, scores):
        if score is not None:
            text = "{}: {:.2f}".format(_cls, score)
        else:
            text = "{}".format(_cls)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1-8,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.savefig(fname, dpi=60, bbox_inches='tight', transparent=True, pad_inches=0)

class VisualizePredictions(keras.callbacks.Callback):
    """VisualizePredictions visualizes predictions from RetinaNet.

    Args:
        model: RetinaNet model to generate the predictions,
        test_images: array-like of images to predict detections for.
        artifact_dir: directory to store images in.
    """
    def __init__(self, test_image, test_boxes, artifact_dir, **kwargs):
        super().__init__(**kwargs)
        self.test_image = test_image
        self.test_boxes = test_boxes
        self.artifact_dir = artifact_dir

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        class_names = ["COTS"] * test_boxes.shape[0]
        scores = [None] * test_boxes.shape[0]

        width, height = test_image.shape[1], test_image.shape[0]

        bbox = test_boxes.numpy()
        bbox[:, 0] *= height
        bbox[:, 1] *= width
        bbox[:, 2] *= height
        bbox[:, 3] *= width

        bbox = swap_xy(bbox)

        visualize_detections(
            test_image,
            bbox,
            class_names,
            scores,
            self.dir_path + "/ground_truth.png"
        )

    @property
    def dir_path(self):
        return self.artifact_dir + "/predictions"

    def on_epoch_end(self, epoch, logs=None):
        test_image = self.test_image
        input_image, ratio = prepare_image(test_image)
        input_image = tf.expand_dims(input_image, axis=0)

        boxes = self.model.inference(input_image)
        path = f"{self.dir_path}/epoch/"
        result = boxes[0].numpy()
        num_detections = detections.valid_detections[0]
        class_names = [
            "COTS" for x in detections.nmsed_classes[0][:num_detections]
        ]
        visualize_detections(
            image,
            detections.nmsed_boxes[0][:num_detections] / ratio,
            class_names,
            detections.nmsed_scores[0][:num_detections],
            f"{self.dir_path}/{epoch}.png"
        )
