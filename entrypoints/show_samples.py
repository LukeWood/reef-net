import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import reef_net
from reef_net.utils import convert_to_corners
from reef_net.utils import convert_to_xywh
from reef_net.utils import swap_xy

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config", "configs/main.py")


def visualize_bounding_boxes(img, bbox, category):
    bbox = swap_xy(bbox)  # Swap_xy makes this go Nan as of now I suppose
    image_shape = img.shape

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    bbox = convert_to_corners(bbox)
    bbox = bbox.numpy()
    for annotation in bbox:
        x1, y1, x2, y2 = annotation.astype(int)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return img


def main(args):
    config = FLAGS.config
    ds, dataset_size = reef_net.loaders.load_reef_dataset(
        config, "custom_csv/train.csv", min_boxes_per_image=5
    )
    ds = ds.shuffle(20)

    (image, bounding_boxes, category) = next(iter(ds.take(100)))
    image, bounding_boxes, category = (
        image.numpy(),
        bounding_boxes.numpy(),
        category.numpy(),
    )
    plt.imshow(image / 255.0)
    plt.axis("off")
    plt.show()
    print("Category", category)
    print("Image size", image.shape)
    print(category)

    image = visualize_bounding_boxes(image, bounding_boxes, category)
    plt.imshow(image / 255.0)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    app.run(main)
