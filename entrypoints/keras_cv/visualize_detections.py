import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from reef_net.preprocessing import resize_and_pad_image

import keras_cv


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    return tf.expand_dims(image, axis=0), ratio


def visualize_detections(
    bounding_box_format,
    image,
    boxes,
    classes,
    scores,
    fname,
    figsize=(24, 16),
    linewidth=1,
    color=[0, 0, 1],
):
    """Visualize Detections"""

    image = np.array(image, dtype=np.uint8)
    boxes = keras_cv.bounding_box.convert_format(
        boxes, source=bounding_box_format, target="xyxy", images=image
    )
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
            y1 - 8,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.savefig(fname, dpi=60, bbox_inches="tight", transparent=True, pad_inches=0)


class VisualizePredictions(keras.callbacks.Callback):
    """VisualizePredictions visualizes predictions from RetinaNet.

    Args:
        model: RetinaNet model to generate the predictions,
        test_images: array-like of images to predict detections for.
        artifact_dir: directory to store images in.
        subdir_name: subdirectory to store images in
    """

    def __init__(
        self,
        bounding_box_format,
        test_image,
        test_boxes,
        artifact_dir,
        subdir_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.test_image = test_image
        self.test_boxes = test_boxes
        self.artifact_dir = artifact_dir
        self.subdir_name = subdir_name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        class_names = ["COTS"] * test_boxes.shape[0]
        scores = [None] * test_boxes.shape[0]

        visualize_detections(
            bounding_box_format,
            test_image,
            test_boxes,
            class_names,
            scores,
            self.dir_path + "/ground_truth.png",
        )

    @property
    def dir_path(self):
        return self.artifact_dir + "/" + self.subdir_name

    def on_epoch_end(self, epoch, logs=None):
        test_image = self.test_image
        input_image, ratio = prepare_image(test_image)
        detections = self.model(input_image)["inference"]

        print('detections.shape before', detections.shape)
        detections = detections[0]
        print('detections.shape after', detections.shape)
        num_detections = detections.shape[0]
        print('num_detections', num_detections)

        if num_detections == 0:
            print('num_detections was 0, not logging inference')
            return
        class_names = ["COTS" for x in range(num_detections)]
        visualize_detections(
            self.bounding_box_format,
            test_image,
            detections[:4] / ratio,
            class_names,
            detections[5],
            f"{self.dir_path}/{epoch}.png",
        )
