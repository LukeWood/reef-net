import cv2
import matplotlib.pyplot as plt
import numpy as np
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import reef_net
from reef_net.preprocess import preprocess_data

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config", "configs/main.py")


def visualize_bounding_boxes(img, annotations, category):
    for annotation in annotations:
        x, y, w, h = annotation.astype(np.float32)
        x1 = (x - w / 2).astype(int)
        x2 = (x + w / 2).astype(int)
        y1 = (y - h / 2).astype(int)
        y2 = (y + h / 2).astype(int)
        if category.size == 0:
            break
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return img


def main(args):
    config = FLAGS.config
    ds, dataset_size = reef_net.loaders.load_reef_dataset(config, min_boxes_per_image=1)
    ds = ds.shuffle(20)
    ds = ds.map(preprocess_data)

    (image, bounding_boxes, category) = next(iter(ds.take(100)))
    image, bounding_boxes, category = (
        image.numpy(),
        bounding_boxes.numpy(),
        category.numpy(),
    )
    print("Category", category)
    print("Image size", image.shape)
    image = visualize_bounding_boxes(image, bounding_boxes, category)
    plt.imshow(image / 255.0)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    app.run(main)
