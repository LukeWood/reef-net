from __future__ import annotations
import cv2
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import reef_net
import wandb
from reef_net.preprocess import preprocess_data
from reef_net.utils import visualize_detections



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
    ds, dataset_size = reef_net.loaders.load_reef_dataset(config, min_boxes_per_image=5)

    ds = ds.shuffle(1)
    # print("Preprocess Data")
    ds = ds.map(preprocess_data)
    # print("Processed")

    (image, bounding_boxes, category) = next(iter(ds.take(1)))
    image, bounding_boxes, category = (
        image.numpy(),
        bounding_boxes.numpy(),
        category.numpy(),
    )

    # image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Temp
    scores = [[0.99] * bounding_boxes.shape[0]]
    scores = np.concatenate(scores).flat
    
    # image = visualize_bounding_boxes(image, bounding_boxes, category)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image / 255.0)
    # plt.axis("off")
    # plt.show()
    # img_h, img_w, _ = image.shape
    # bounding_boxes[:, 0] = bounding_boxes[:, 0] * img_h
    # bounding_boxes[:, 1] = bounding_boxes[:, 1] * img_w
    # bounding_boxes[:, 2] = bounding_boxes[:, 2] * img_h
    # bounding_boxes[:, 3] = bounding_boxes[:, 3] * img_w

    visualize_detections(image, bounding_boxes, category, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1])


if __name__ == "__main__":
    app.run(main)
